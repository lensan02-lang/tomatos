import copy
import json
import logging
import os
import sys
from functools import partial
from time import perf_counter

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from alive_progress import alive_it
from jaxopt import OptaxSolver
import pyhf

import tomatos.batcher
import tomatos.constraints
import tomatos.histograms
import tomatos.nn
import tomatos.pipeline
import tomatos.solver
import tomatos.utils
import tomatos.workspace
import tomatos.select


def binary_cross_entropy_logits(logits, labels, weights=None):
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    if weights is None:
        return jnp.mean(loss)
    weights = jnp.asarray(weights)
    return jnp.sum(weights * loss) / (
        jnp.sum(jnp.abs(weights)) + 1e-15
    )

def bce(ones, zeros, ones_weights=None, zeros_weights=None):
    labels = jnp.concatenate(
        [
            jnp.ones_like(ones),
            jnp.zeros_like(zeros),
        ]
    )
    logits = jnp.concatenate((ones, zeros))
    if ones_weights is None and zeros_weights is None:
        return binary_cross_entropy_logits(
            logits,
            labels,
        )
    weights = jnp.concatenate(
        (
            jnp.ones_like(ones)
            if ones_weights is None
            else ones_weights,
            jnp.ones_like(zeros)
            if zeros_weights is None
            else zeros_weights,
        )
    )
    return binary_cross_entropy_logits(
        logits,
        labels,
        weights,
    )

def get_epoch_keys(metrics_dict, config):

    epoch_keys = [
        key for key in metrics_dict.keys()
        if key.startswith("epoch_")
    ]

    if not epoch_keys:
        raise ValueError("No epoch keys found")

    numeric_epoch_keys = [
        k for k in epoch_keys
        if k.split("_")[1].isdigit()
    ]

    latest_key = max(
        numeric_epoch_keys,
        key=lambda x: int(x.split("_")[1])
    )

    best_key = (
        "epoch_best"
        if "epoch_best" in metrics_dict
        else None
    )

    predefined_epoch_key = None

    if getattr(config, "pretrain_epochnumber", None) is not None:

        for k in numeric_epoch_keys:

            epoch_num = int(k.split("_")[1])

            if epoch_num == config.pretrain_epochnumber:
                predefined_epoch_key = k
                break

    return latest_key, best_key, predefined_epoch_key

def log_nn_output(metrics, opt_pars, data, scale, config, step_i):
    if "nn" not in opt_pars:
        logging.warning("log_nn_output: no nn parameters present in opt_pars")
        return None

    nn_output = tomatos.histograms.get_nn_output(
        opt_pars,
        data,
        config.nn_arch,
        config.nn_inputs_idx_end,
    )
    metrics["nn_output_mean"] = float(np.mean(nn_output))
    metrics["nn_output_std"] = float(np.std(nn_output))
    metrics["nn_output_min"] = float(np.min(nn_output))
    metrics["nn_output_max"] = float(np.max(nn_output))

    logging.info(f"nn output - spread: {metrics['nn_output_max']-metrics['nn_output_min']:.6f}")

    sig = nn_output[config.samples.index(config.signal_sample), :]
    bkg = nn_output[config.samples.index("bkg"), :]
    sig = np.asarray(sig)
    bkg = np.asarray(bkg)
    metrics["nn_output_signal_mean"] = float(np.mean(sig))
    metrics["nn_output_signal_std"] = float(np.std(sig))
    metrics["nn_output_bkg_mean"] = float(np.mean(bkg))
    metrics["nn_output_bkg_std"] = float(np.std(bkg))
    
    gap = metrics["nn_output_signal_mean"] - metrics["nn_output_bkg_mean"]
    logging.info(f"separation gap (sig_mean - bkg_mean): {gap:.6f}")
    bkg = np.asarray(bkg)
    sig = np.asarray(sig)

    mask_low  = bkg < 0.3
    mask_mid  = (bkg > 0.3) & (bkg < 0.7)
    mask_high = bkg > 0.7

    weights_bkg = data[config.samples.index("bkg"), :, config.weight_idx]
    weights_sig = data[config.samples.index(config.signal_sample), :, config.weight_idx]

    #sel_weights = tomatos.select.events(data, config, )

    btag_1_bkg = data[config.samples.index("bkg"), :, config.vars.index("bool_btag_1")]
    btag_2_bkg = data[config.samples.index("bkg"), :, config.vars.index("bool_btag_2")]
    btag_1_sig = data[config.samples.index(config.signal_sample), :, config.vars.index("bool_btag_1")]
    btag_2_sig = data[config.samples.index(config.signal_sample), :, config.vars.index("bool_btag_2")]

    mask_btag1_bkg = btag_1_bkg == 1 
    mask_btag2_bkg = btag_2_bkg == 1
    mask_btag1_sig = btag_1_sig == 1
    mask_btag2_sig = btag_2_sig == 1

    btag = []
    btag.append(bkg[mask_btag1_bkg])
    btag.append(bkg[mask_btag2_bkg])
    btag.append(sig[mask_btag1_sig])
    btag.append(sig[mask_btag2_sig])

    btag_weights = []
    btag_weights.append(weights_bkg[mask_btag1_bkg])
    btag_weights.append(weights_bkg[mask_btag2_bkg])
    btag_weights.append(weights_sig[mask_btag1_sig])
    btag_weights.append(weights_sig[mask_btag2_sig])

    scale_bkg = scale[0]
    scale_sig = scale[1]
    scaled_weights = [
    btag_weights[0] * scale_bkg,
    btag_weights[1] * scale_bkg,
    btag_weights[2] * scale_sig,
    btag_weights[3] * scale_sig,
    ]

    fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    sharex=True,
    gridspec_kw={"height_ratios": [3,1]},
    figsize=(8,8)
    )

    bins = np.linspace(0, 1, 41)

    # -----------------------------------
    # Background histograms
    # -----------------------------------

    hist_bkg1, bins, _ = ax1.hist(
        bkg[mask_btag1_bkg],
        bins=bins,
        weights=weights_bkg[mask_btag1_bkg] * scale_bkg,
        histtype="stepfilled",
        alpha=0.3,
        label="bkg btag1"
    )

    hist_bkg2, _, _ = ax1.hist(
        bkg[mask_btag2_bkg],
        bins=bins,
        weights=weights_bkg[mask_btag2_bkg] * scale_bkg,
        histtype="stepfilled",
        alpha=0.3,
        label="bkg btag2"
    )

    # -----------------------------------
    # Signal histograms
    # -----------------------------------

    hist_sig1,_,_=ax1.hist(
        sig[mask_btag1_sig],
        bins=bins,
        weights=weights_sig[mask_btag1_sig] * scale_sig,
        histtype="step",
        linewidth=2,
        label="sig btag1"
    )

    hist_sig2,_,_=ax1.hist(
        sig[mask_btag2_sig],
        bins=bins,
        weights=weights_sig[mask_btag2_sig] * scale_sig,
        histtype="step",
        linewidth=2,
        label="sig btag2"
    )

    # -----------------------------------
    # Main plot styling
    # -----------------------------------

    ax1.set_title("NN output distribution by btag category")
    ax1.set_ylabel("Weighted events")
    ax1.set_yscale("log")
    ax1.legend()

    # -----------------------------------
    # letzer bin ratio plot
    # -----------------------------------

    ratio = hist_bkg2 / (hist_sig2 + 1e-8)

    centers = 0.5 * (bins[1:] + bins[:-1])

    ax2.step(
        centers,
        ratio,
        where="mid"
    )

    ax2.axhline(1.0, linestyle="--")

    ax2.set_xlabel("NN score")
    ax2.set_ylabel("btag2bkg / btag2sig")

    ax2.set_ylim(0, 5)

    fig.tight_layout()

    plt.savefig(
        config.plot_path + f"nn_score_ratio_{step_i}.pdf"
    )

    plt.close()

def log_abcd_closure(config, opt_pars, data, scale, step_i):
    hists = tomatos.pipeline.make_hists(
        opt_pars, data, config, scale, filter_return_hists=False, step=step_i
    )
    bkg_sr1 = np.array(hists["SR_btag_1"]["bkg"]["NOSYS"])
    bkg_sr2 = np.array(hists["SR_btag_2"]["bkg"]["NOSYS"])

    sum1, sum2 = bkg_sr1.sum(), bkg_sr2.sum()
    if sum1 < 1e-10 or sum2 < 1e-10:
        return

    bkg_sr1_norm = bkg_sr1 / sum1
    bkg_sr2_norm = bkg_sr2 / sum2

    bins = config.bins
    centers = 0.5 * (bins[1:] + bins[:-1])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(8, 6),
    )
    ax1.stairs(bkg_sr2_norm, bins, label="bkg SR_btag_2", color="steelblue")
    ax1.stairs(bkg_sr1_norm, bins, label="bkg SR_btag_1 (normalized)", color="tomato", linestyle="--")
    ax1.set_ylabel("Normalized Events")
    ax1.set_title(f"ABCD Shape Closure — step {step_i}")
    ax1.legend()

    ratio = np.where(bkg_sr2_norm > 1e-8, bkg_sr1_norm / bkg_sr2_norm, np.nan)
    ax2.step(centers, ratio, where="mid", color="black")
    ax2.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
    ax2.set_xlabel("NN Score")
    ax2.set_ylabel("SR_btag_1 / SR_btag_2")
    ax2.set_ylim(0.5, 1.5)

    fig.tight_layout()
    plt.savefig(config.plot_path + f"abcd_closure_{step_i:05d}.pdf")
    plt.close()


def log_bw(metrics, opt_pars):
    metrics["bw"] = opt_pars["bw"]
    logging.info(f"bw: {opt_pars['bw']}")


def log_cuts(config, opt_pars, metrics, infer_metrics_i):
    for var, cut_dict in config.opt_cuts.items():
        cut_var = f"cut_{var}"
        opt_cut = tomatos.utils.inverse_min_max_scale(
            config, opt_pars[cut_var], cut_dict["idx"]
        )
        logging.info(f"{cut_var}: {opt_cut}")
        if cut_var not in metrics:
            metrics[cut_var] = []
        metrics[cut_var] = opt_cut
        infer_metrics_i[cut_var] = opt_cut


def log_kde(config, metrics, opt_pars, train_data, train_sf, hists, bins, step=0):
    kde = sample_kde_distribution(
        config=config,
        opt_pars=opt_pars,
        data=train_data,
        scale=train_sf,
        hists=hists,
        bins=bins,
        step=step,
    )
    for key, h in kde.items():
        metrics["kde_" + key] = h


def log_hists(config, metrics, test_hists, hists):
    for h_key, h in hists.items():
        metrics["h_" + h_key] = h
    for h_key, h in test_hists.items():
        metrics["h_" + h_key + "_test"] = h

        # nominal hists
    logging.info("--- Nominal (binned KDE) ---")
    for key, h in hists.items():
        if config.nominal in key and not "STAT" in key:
            logging.info(f"{key.ljust(25)}: {h}")
    logging.info("--- Uncertainty (binned KDE) ---")
    for key, h in hists.items():
        if "1UP" in key or "1DOWN" in key:
            logging.info(f"{key.ljust(25)}: {h}")


def log_bins(config, metrics, bins, infer_metrics_i):
    scaled_bins = (
        tomatos.utils.inverse_min_max_scale(config, np.copy(bins), config.cls_var_idx)
        if config.objective == "cls_var"
        else bins
    )
    metrics["bins"] = scaled_bins
    infer_metrics_i["bins"] = scaled_bins

    if config.include_bins:
        logging.info(f"{'bins'.ljust(25)}: {scaled_bins}")

    return bins


def rescale_kde(config, hist, kde, bins):

    # need to upscale sampled kde hist as it is a very fine binned version of
    # the histogram, use the largest bin for it,
    # NB: this is an approximation, only works properly for the largest bin

    # use the largest bin of a binned kde hist
    max_bin_idx = np.argmax(hist)
    max_bin_edges = np.array([bins[max_bin_idx], bins[max_bin_idx + 1]])
    # integrate histogram for this bin
    hist_x_width = np.diff(max_bin_edges)
    hist_height = hist[max_bin_idx]
    area_hist = hist_x_width * hist_height

    # integrate kde for this bin
    kde_indices = (max_bin_edges * config.kde_sampling).astype(int)
    kde_heights = kde[kde_indices[0] : kde_indices[1]]
    kde_dx = 1 / config.kde_sampling
    area_kde = np.sum(kde_dx * kde_heights)

    scale_factor = area_hist / area_kde
    kde_scaled = kde * scale_factor

    return kde_scaled


def sample_kde_distribution(
    config,
    opt_pars,
    data,
    scale,
    hists,
    bins,
    step=0,
):
    # get kde distribution by sampling with a many bin histogram
    # enough to get kde only from the nominal ones
    sample_indices = np.arange(len(config.samples))
    nominal_data = data[sample_indices, :, :]
    # make a custom config
    kde_config = copy.deepcopy(config)
    kde_config.bins = config.kde_bins
    kde_config.include_bins = False

    # to also collect the background estimate
    kde_dist = tomatos.pipeline.make_hists(
        opt_pars,
        nominal_data,
        kde_config,
        scale,
        filter_return_hists=True,
        step=step,
    )

    kde_dist = {
        h_key: rescale_kde(config, hists[h_key], kde_dist[h_key], bins)
        for h_key in kde_dist
    }

    return kde_dist


def log_sharp_hists(
    opt_pars,
    train_data,
    config,
    train_sf,
    hists,
    metrics,
    step=0,
):
    # actually might be enough to compare to test hists, depends a bit on the
    # uncertainty behavior...

    # sharp evaluation train data hists
    sharp_hists = tomatos.pipeline.make_hists(
        opt_pars,
        train_data,
        config,
        train_sf,
        validate_only=True,  # sharp hists
        filter_return_hists=True,
        step=step,
    )
    logging.info("--- Nominal (Sharp hist) ---")
    for (h_key, h) in hists.items():
        if config.nominal in h_key and not "STAT" in h_key:
            sharp_h = sharp_hists[h_key]
            logging.info(f"{h_key.ljust(25)}: {sharp_h}")
    logging.info("--- Uncertainty (Sharp hist) ---")
    for (h_key, h) in hists.items():
        if "1UP" in h_key or "1DOWN" in h_key:
            sharp_h = sharp_hists[h_key]
            logging.info(f"{h_key.ljust(25)}: {sharp_h}")

    logging.info("--- Nominal (binned KDE) / (Sharp hist) ---")
    for (h_key, h) in hists.items():        
        if config.nominal in h_key and not "STAT" in h_key:           
            sharp_h = sharp_hists[h_key]            
            # hist approx ratio - protect against division by zero            
            metrics["h_" + h_key + "_sharp"] = sharp_h            
            # Use np.where to avoid inf/nan from division by very small values
            ratio = np.where(
                sharp_h > 1e-6,  # threshold to avoid division by zero
                h / sharp_h,
                0.0,  # or np.nan, depending on how you want to handle this case
            )
            logging.info(f"{h_key.ljust(25)}: {ratio}")


def do_metrics_exist(config):
    if os.path.exists(config.metrics_file_path) and not config.debug:
        user_input = input(
            f"{config.metrics_file_path} exists. \n"
            "Seems like you trained this already \n"
            "Overwrite and Proceed? (y/n):"
        )
        if user_input.lower() != "y":
            logging.info("OK, Bye!")
            sys.exit(1)


def init_metrics(config, metrics):
    with h5py.File(config.metrics_file_path, "w") as h5f:
        for key, value in metrics.items():
            if isinstance(value, float):
                shape = (config.num_steps,)
                dtype = "f4"
            elif isinstance(value, int):
                shape = (config.num_steps,)
                dtype = "i"
            elif isinstance(value, list):
                shape = (config.num_steps, len(value))
                dtype = "f4"
            h5f.create_dataset(key, shape=shape, dtype=dtype, compression="gzip")
            h5f[key][0] = value


def write_metrics(config, metrics, i):
    metrics = tomatos.utils.to_python_lists(metrics)
    if i == 0:
        init_metrics(config, metrics)
    else:
        with h5py.File(config.metrics_file_path, "r+") as h5f:
            for key, value in metrics.items():
                dataset = h5f[key]
                if isinstance(value, (float, int)):
                    dataset[i] = value
                else:
                    dataset[i, :] = value


def save_model(
    i,
    test_loss,
    best_test_loss,
    config,
    opt_pars,
    infer_metrics,
    infer_metrics_i,
):
    # pick best training and save
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        infer_metrics["epoch_best"] = infer_metrics_i
        infer_metrics["epoch_best"]["epoch"] = i
        model = eqx.combine(opt_pars["nn"], config.nn_arch)
        eqx.tree_serialise_leaves(config.model_path + "epoch_best.eqx", model)
    # save every 10th model to file
    if i % 10 == 0 and i != 0:
        epoch_name = f"epoch_{i:005d}"
        infer_metrics[epoch_name] = infer_metrics_i
        model = eqx.combine(opt_pars["nn"], config.nn_arch)
        eqx.tree_serialise_leaves(config.model_path + epoch_name + ".eqx", model)

    if i == (config.num_steps - 1):
        # save infer metrics
        with open(config.infer_metrics_file_path, "w") as file:
            json.dump(tomatos.utils.to_python_lists(infer_metrics), file)