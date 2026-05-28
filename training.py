import copy
import json
import logging
import sys
from functools import partial
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from alive_progress import alive_it
from jaxopt import OptaxSolver
import os

import tomatos.batcher
import tomatos.constraints
import tomatos.histograms
import tomatos.nn
import tomatos.pipeline
import tomatos.solver
import tomatos.train_utils
import tomatos.utils
import tomatos.workspace

def load_pretrained_params(config, opt_pars):

    if not (
        hasattr(config, "pretrain_infer_metrics")
        and config.pretrain_infer_metrics
    ):
        return opt_pars

    with open(config.pretrain_infer_metrics, "r") as f:
        pretrained_metrics = json.load(f)

    latest_epoch_key, best_key, predefined_epoch_key = (
        tomatos.train_utils.get_epoch_keys(
            pretrained_metrics,
            config
        )
    )

    # choose which epoch to load
    selected_key = predefined_epoch_key or best_key

    if predefined_epoch_key is not None:
        selected_key = predefined_epoch_key

    elif best_key is None:
        selected_key = latest_epoch_key

    print(
        f"Loading pretrained params from {selected_key}"
    )

    best_params = pretrained_metrics[selected_key]

    print(f"Pretrained params: {best_params}")

    # bins
    if "bins" in best_params and config.include_bins:

        scaled_bins = np.array(best_params["bins"])
        opt_pars["bins"] = scaled_bins[1:-1]

    # cuts
    if config.include_cuts:

        for var in config.opt_cuts:

            var_idx = config.vars.index(var)

            scale = config.scaler_scale[var_idx]
            shift = config.scaler_min[var_idx]

            cut_key = f"cut_{var}"

            if cut_key in best_params:

                scaled_cut = best_params[cut_key]

                unscaled_cut = (
                    scaled_cut * scale + shift
                )

                opt_pars[f"cut_{var}"] = unscaled_cut

            elif f"cut_{var}_center" in best_params:

                opt_pars[f"cut_{var}_center"] = (
                    best_params[f"cut_{var}_center"]
                )

                opt_pars[f"cut_{var}_logwidth"] = (
                    best_params[f"cut_{var}_logwidth"]
                )

    # bw
    if "bw" in best_params:
        opt_pars["bw"] = best_params["bw"]
    
    return opt_pars

def init_opt_pars(config, nn_pars):

    # build opt_pars
    opt_pars = {}
    opt_pars["nn"] = nn_pars
    opt_pars["bw"] = config.bw_init
    if config.include_bins:
        # exclude boundaries
        opt_pars["bins"] = config.bins[1:-1]

    for key in config.opt_cuts:
        var_idx = config.vars.index(key)
        config.opt_cuts[key]["idx"] = var_idx
        init = config.opt_cuts[key]["init"]
        init *= config.scaler_scale[var_idx]
        init += config.scaler_min[var_idx]
        opt_pars["cut_" + key] = init

    return opt_pars


def train_init(config):
    # init nn and opt pars
    key = jax.random.PRNGKey(0)
    nn_model = tomatos.nn.NeuralNetwork(n_features=config.nn_inputs_idx_end)
    # split model into parameters to optimize and the nn architecture
    if hasattr(config, "pretrain_params") and config.pretrain_params:
        model_path = os.path.join(config.pretrain_params,f"epoch_00460.eqx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Loading pretrained NN from {model_path}")
        nn_model = eqx.tree_deserialise_leaves(model_path, nn_model)
        print("NN-Model:")
        print(nn_model)
    nn_pars, nn_arch = eqx.partition(nn_model, eqx.is_array)
    config.nn_arch = nn_arch

    # get preprocess md
    with open(config.preprocess_md_file_path, "r") as json_file:
        config.preprocess_md = json.load(json_file)
    # for unscaling of vars
    config.scaler_scale = np.array(config.preprocess_md["scaler_scale"])
    config.scaler_min = np.array(config.preprocess_md["scaler_min"])

    opt_pars = init_opt_pars(config, nn_pars)
        
    # Load pretrained parameters if specified
    opt_pars = load_pretrained_params(config, opt_pars)
    
    # batcher
    batch = {}
    for split in ["train", "valid", "test"]:
        batch[split] = tomatos.batcher.get_generator(config, split)

    # solver
    solver = tomatos.solver.setup(config, opt_pars)
    train_data, train_sf = next(batch["train"])
    state = solver.init_state(
        opt_pars,
        data=train_data,
        config=config,
        scale=train_sf,
    )

    best_test_loss = np.inf

    return solver, state, opt_pars, batch, best_test_loss


def evaluate_losses(opt_pars, config, batch, step=0):
    """Evaluates validation and test losses."""
    valid_data, valid_sf = next(batch["valid"])
    valid_loss, valid_hists = tomatos.pipeline.loss_fn(
        opt_pars, valid_data, config, valid_sf, validate_only=True, step=step
    )

    test_data, test_sf = next(batch["test"])
    test_loss, test_hists = tomatos.pipeline.loss_fn(
        opt_pars, test_data, config, test_sf, validate_only=True, step=step
    )

    return valid_loss, valid_hists, test_loss, test_hists


def run(config):
    # expensive in here is evaluate_losses and solver.update

    solver, state, opt_pars, batch, best_test_loss = train_init(config)

    metrics = {}
    # this holds optimization params like cuts for epochs used for deployment
    infer_metrics = {}
    # don't overwrite by mistake
    tomatos.train_utils.do_metrics_exist(config)

    # one step is one batch (not epoch)
    for i in alive_it(range(config.num_steps)):
        start = perf_counter()
        logging.info(f"step {i}: loss={config.objective}")

        # this holds optimization params like cuts per batch, for deployment
        infer_metrics_i = {}

        # this has to be here
        # since the optaxsolver holds step i-1, train evaluation is expensive
        valid_loss, valid_hists, test_loss, test_hists = evaluate_losses(
            opt_pars, config, batch, step=i
        )
        metrics["train_loss"] = state.value
        metrics["valid_loss"] = valid_loss
        metrics["test_loss"] = test_loss

        # gradient update
        train_data, train_sf = next(batch["train"])
        opt_pars, state = solver.update(
            opt_pars,
            state,
            data=train_data,
            config=config,
            scale=train_sf,
            step=i,
        )
        # apply limitations
        opt_pars = tomatos.constraints.opt_pars(config, opt_pars)

        ###### excessive logging, turn off as you please ######
        hists = state.aux
        bins = (
            np.array([0, *opt_pars["bins"], 1]) if config.include_bins else config.bins
        )
        tomatos.train_utils.log_hists(config, metrics, test_hists, hists)
        tomatos.train_utils.log_kde(
            config, metrics, opt_pars, train_data, train_sf, hists, bins, step=i
        )

        if "cls" in config.objective or "bce" in config.objective:
            tomatos.train_utils.log_sharp_hists(
                opt_pars, train_data, config, train_sf, hists, metrics, step=i
            )
            if i%10 == 0:
                tomatos.train_utils.log_nn_output(metrics, opt_pars, train_data, train_sf, config, i)
                tomatos.train_utils.log_abcd_closure(config, opt_pars, train_data, train_sf, i)
            tomatos.train_utils.log_bins(config, metrics, bins, infer_metrics_i)
            tomatos.train_utils.log_cuts(config, opt_pars, metrics, infer_metrics_i)
            tomatos.train_utils.log_bw(metrics, opt_pars)
        tomatos.train_utils.save_model(
            i,
            test_loss,
            best_test_loss,
            config,
            opt_pars,
            infer_metrics,
            infer_metrics_i,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            metrics["best_test_batch"] = i

        tomatos.train_utils.write_metrics(config, metrics, i)

        # if your memory explodes, clear jax compilation caches
        tomatos.utils.clear_caches(config)

        end = perf_counter()
        logging.info(f"train loss: {state.value}")
        logging.info(f"test loss: {test_loss}")
        logging.info(f"update took {end-start:.4f}s")
        logging.info("\n")

    logging.info("Training Done!")
    return