import pprint
from functools import partial

import jax
import jax.numpy as jnp
import neos
import pyhf

import tomatos.constraints
import tomatos.histograms
import tomatos.select
import tomatos.train_utils
import tomatos.utils
import tomatos.workspace
from tomatos.histograms import get_nn_output


def make_hists(
    pars, data, config, scale, validate_only=False, filter_return_hists=False, step=0
):
    # event manipulations are done via weights to the base weights
    base_weights = data[:, :, config.weight_idx]
    cut_weights = tomatos.select.cuts(pars, data, config, validate_only, step)
    # apply cuts
    base_weights = jnp.multiply(base_weights, cut_weights)
    # get event selections
    sel_weights = tomatos.select.events(data, config, base_weights)
    # fill
    hists = tomatos.histograms.fill_hists(
        pars, data, config, sel_weights, scale, validate_only
    )
    # calculate additional hists based on existing hists
    hists = tomatos.workspace.hist_transforms(hists, validate_only)
    # flatten and filter if desired
    hists = tomatos.utils.filter_hists(config, hists) if filter_return_hists else hists
    # plot hists to see if bkg estimation is working as expected
    #tomatos.plotting.plot_bkg_estimate(config, hists)
    return hists


def loss_fn(
    pars,  # OptaxSolver expects opt_pars as first arg
    data,
    config,
    scale,
    validate_only=False,
    filter_return_hists=True,
    step = 0,
):
    # the main reason why not everything in here is jitted, is that the
    # config is not a jax compatible type (pytree), this will be a bit tedious
    # as in particular you have to get rid of all strings
    nn_output = get_nn_output(
            pars,
            data,
            config.nn_arch,
            config.nn_inputs_idx_end,
        )
        
    hists = make_hists(pars, data, config, scale, validate_only, step=step)

    signal_idx = config.samples.index(config.signal_sample)
    bkg_idx = config.samples.index("bkg")
    sig = nn_output[signal_idx, :]
    bkg = nn_output[bkg_idx, :]

    base_weights = data[:, :, config.weight_idx]
    cut_weights = tomatos.select.cuts(pars, data, config, validate_only, step)
    sig_weights = base_weights[signal_idx, :] * cut_weights[signal_idx, :]
    bkg_weights = base_weights[bkg_idx, :] * cut_weights[bkg_idx, :]
    # use abs sum to avoid NaN when negative MC weights partially cancel
    sig_weights_bce = sig_weights / (jnp.sum(jnp.abs(sig_weights)) + 1e-8)
    bkg_weights_bce = bkg_weights / (jnp.sum(jnp.abs(bkg_weights)) + 1e-8)

    if "bce" in config.objective:
        loss_value = tomatos.train_utils.bce(ones=sig, zeros=bkg, ones_weights=sig_weights_bce, zeros_weights=bkg_weights_bce)

    if config.objective == "cls_nn" or config.objective == "cls_var":
        bce_loss = tomatos.train_utils.bce(
            ones=sig,
            zeros=bkg,
            ones_weights=sig_weights_bce,
            zeros_weights=bkg_weights_bce,
        )
        # warmup: pure BCE during training (not validate_only) to avoid NaN
        # from degenerate histograms with untrained NN; validation always uses CLs
        if step < config.bce_warmup_steps:
            loss_value = bce_loss
        else:
            model, hists = tomatos.workspace.pyhf_model(hists, config, validate_only=validate_only)
            cls = neos.loss_from_model(model, loss="cls")
            # fall back to BCE when CLs is NaN (unstable pyhf fit);
            # zero_nans() in the optimizer zeroes the NaN gradient from cls
            loss_value = jnp.where(jnp.isnan(cls), bce_loss, cls)

    if not validate_only:
            loss_value = tomatos.constraints.penalize_loss(loss_value, hists)

    # flatten and reduces to the configured filter
    hists = tomatos.utils.filter_hists(config, hists) if filter_return_hists else hists
    return loss_value, hists