import json
import logging

import h5py
import numpy as np
import uproot
from alive_progress import alive_it
from sklearn.preprocessing import MinMaxScaler

import tomatos.utils


def init_2d_data(config):
    with h5py.File(config.preprocess_files["data"], "w") as f:
        for sample_sys in config.sample_sys:
            f.create_dataset(
                name=sample_sys,
                shape=(0, len(config.vars)),
                maxshape=(
                    None,  # allows resize, makes appending easy, instead of idx tracking
                    len(config.vars),
                ),
                compression="lzf",
                dtype="f4",
                chunks=True,  # autochunking, since resizing
            )


def fill_2d_and_find_scale(
    config,
    sample_sys,
    data,
    scaler,
):
    # transform dict data into 2d array of dimension (events, vars)
    data_2d = np.array([data[var] for var in config.vars]).T
    # iteratively update min max for scaling of train vars
    scaler.partial_fit(data_2d[:, : config.nn_inputs_idx_end])
    with h5py.File(config.preprocess_files["data"], "r+") as f:
        ds = f[sample_sys]
        old_shape = ds.shape
        new_shape = (old_shape[0] + data_2d.shape[0], old_shape[1])
        # resize array on disk
        ds.resize(new_shape)
        # append on disk
        ds[old_shape[0] :] = data_2d

    return


def fill_2d_data(config, scaler):
    for sample_sys, path in alive_it(config.sample_files_dict.items()):
        with uproot.open(path) as file:
            tree = file[config.tree_name]
            for data in tree.iterate(step_size=config.chunk_size, library="np"):
                #######
                # preselection could be here
                # ...transfer generate test files code
                #######
                fill_2d_and_find_scale(config, sample_sys, data, scaler)


def init_preprocess_md(config, max_events):
    preprocess_md = {
        k: {
            "ratio": getattr(config, f"{k}_ratio"),
            "events": -1,
            "scale_factor": np.ones(len(config.sample_sys)),
        }
        for k in ["train", "valid", "test"]
    }
    # Calculate split indices
    train_end = round(max_events * preprocess_md["train"]["ratio"])
    valid_end = train_end + round(max_events * preprocess_md["valid"]["ratio"])

    preprocess_md["train"]["idx_bounds"] = (0, train_end)
    preprocess_md["valid"]["idx_bounds"] = (train_end, valid_end)
    preprocess_md["test"]["idx_bounds"] = (valid_end, max_events)

    # Assign event counts and log results
    for split in ["train", "valid", "test"]:
        start_idx, end_idx = preprocess_md[split]["idx_bounds"]
        n_events = end_idx - start_idx
        preprocess_md[split]["events"] = n_events

    return preprocess_md


def get_max_events(config):
    max_events = 0
    with h5py.File(config.preprocess_files["data"], "r") as f:
        for sample_sys in config.sample_sys:
            n = f[sample_sys].shape[0]
            if n > max_events:
                max_events = n
    return max_events


def init_splits(config, preprocess_md):
    for split in ["train", "valid", "test"]:
        with h5py.File(config.preprocess_files[split], "w") as f_split:
            f_split.create_dataset(
                name="stack",
                shape=(
                    len(config.sample_sys),
                    preprocess_md[split]["events"],
                    len(config.vars),
                ),
                compression="lzf",
                dtype="f4",
                chunks=(
                    len(config.sample_sys),
                    np.minimum(
                        config.chunk_size,
                        preprocess_md[split]["events"] / config.n_chunk_combine,
                    ),  # / 2 see batcher
                    len(config.vars),
                ),
            )


def fill_splits(config, max_events, preprocess_md, scaler):
    # upsampling is straightforward and intuitive to avoid a class imbalance
    # makes particular sense for jax as it likes predefined array sizes,
    # also unifies workflow a lot

    # these are just file handles
    with (
        h5py.File(config.preprocess_files["data"], "r") as f_data,
        h5py.File(config.preprocess_files["train"], "r+") as f_train,
        h5py.File(config.preprocess_files["valid"], "r+") as f_valid,
        h5py.File(config.preprocess_files["test"], "r+") as f_test,
    ):
        np.random.seed(1)
        file = {
            "data": f_data,
            "train": f_train,
            "valid": f_valid,
            "test": f_test,
        }

        # Generate one shared permutation per sample (not per sample_sys).
        # All systematic variations of the same sample use the same perm so
        # that position k in the HDF5 stack always corresponds to the same
        # physical event for NOSYS and e.g. JET_PT_1UP. Without this,
        # independent shuffles make each batch a random comparison between
        # unrelated events, causing systematic histograms to fluctuate above
        # and below nominal purely by chance.
        sample_perm = {
            sample: np.random.permutation(max_events)
            for sample in config.samples
        }

        # upsampling, shuffling, scaling nn inputs
        # having them all together here avoids multiple IO
        for i, sample_sys in alive_it(enumerate(config.sample_sys)):
            # quick and cheap for now, avoids random idx disk IO
            # ds=(max_events=1e7, config.vars=20) ~ 1.6 GB for float32
            ds = file["data"][sample_sys][:]
            n = ds.shape[0]
            upsample_sf = n / max_events
            sample, _ = config.sample_sys_dict[sample_sys]
            # perm % n maps [0, max_events) -> [0, n) cyclically, combining
            # shuffle + upsample in one step with a coordinated index shared
            # across all sys of the same sample
            ds = ds[sample_perm[sample] % n]
            # min max scale nn input vars for this sample_sys
            ds[:, : config.nn_inputs_idx_end] = scaler.transform(
                ds[:, : config.nn_inputs_idx_end]
            )
            # write
            for split in ["train", "valid", "test"]:
                start_idx, end_idx = preprocess_md[split]["idx_bounds"]
                out = ds[start_idx:end_idx]
                split_sf = 1 / preprocess_md[split]["ratio"]
                # account for resampling, splitting, and k-folding, such that
                # except for stats the hist yields in each split are the same
                preprocess_md[split]["scale_factor"][i] *= (
                    upsample_sf * split_sf * config.k_fold_sf
                )
                file[split]["stack"][i, :] = out


def run(config):
    # create data.h5 with 2d data structure (n_events, config.vars)
    init_2d_data(config)
    # scaler for nn inputs
    scaler = MinMaxScaler()
    # fill data.h5 from trees
    logging.info("Prepare Data Array")
    fill_2d_data(config, scaler)
    # scale to sample with most events
    max_events = get_max_events(config)
    preprocess_md = init_preprocess_md(config, max_events)
    # create train.h5, valid.h5, test.h5 with according event_sizes
    init_splits(config, preprocess_md)
    logging.info("Fill Splits")
    fill_splits(config, max_events, preprocess_md, scaler)

    for split in ["train", "valid", "test"]:
        logging.info(f"{split} events: {preprocess_md[split]['events']}")

    # save scaling
    preprocess_md["scaler_min"] = scaler.min_
    preprocess_md["scaler_scale"] = scaler.scale_
    with open(config.preprocess_md_file_path, "w") as json_file:
        json.dump(tomatos.utils.to_python_lists(preprocess_md), json_file, indent=4)