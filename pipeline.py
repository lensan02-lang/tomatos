import pprint
from functools import partial

import jax.numpy as jnp
import neos
import pyhf

import tomatos.constraints
import tomatos.histograms
import tomatos.select
import tomatos.train_utils
import tomatos.utils
import tomatos.workspace
from tomatos.histograms import get_nn_output, get_nn_output_training


def make_hists(
    pars,
    data,
    config,
    scale,
    validate_only=False,
    filter_return_hists=False,
    step=0,
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
        pars, data, config, sel_weights, scale, validate_only, step=step,
    )
    # calculate additional hists based on existing hists
    hists = tomatos.workspace.hist_transforms(
        hists, config, validate_only
    )
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
    step=0,
):
    # the main reason why not everything in here is jitted, is that the
    # config is not a jax compatible type (pytree), this will be a bit tedious
    # as in particular you have to get rid of all strings
    if validate_only:
        nn_output = get_nn_output(pars, data, config.nn_arch, config.nn_inputs_idx_end)
    else:
        nn_output = get_nn_output_training(pars, data, config.nn_arch, config.nn_inputs_idx_end)

    hists = make_hists(
        pars, data, config, scale, validate_only, step=step,
    )

    signal_idx = config.samples.index(config.signal_sample)
    bkg_idx = config.samples.index("bkg")
    sig = nn_output[signal_idx, :]
    bkg = nn_output[bkg_idx, :]

    base_weights = data[:, :, config.weight_idx]
    cut_weights = tomatos.select.cuts(pars, data, config, validate_only, step)
    all_sel_weights = tomatos.select.events(data, config, base_weights * cut_weights)

    def region_norm(w):
        return w / (jnp.sum(jnp.abs(w)) + 1e-8)

    # signal only from SR (ZH→νν bb has negligible MC weight in CR/VR,
    # region_norm would amplify those ghost events and confuse the NN)
    sig_weights_bce = config.signal_bce_weight * region_norm(
        all_sel_weights["SR_btag_2"][signal_idx, :]
    )
    # background from all regions for more statistics; signal stays SR_btag_2-only
    # because ZH→ννbb has negligible MC weight in CR/VR (region_norm would amplify ghost events)
    bkg_weights_bce = (
        region_norm(all_sel_weights["SR_btag_2"][bkg_idx, :]) +
        region_norm(all_sel_weights["SR_btag_1"][bkg_idx, :]) +
        region_norm(all_sel_weights["VR_btag_2"][bkg_idx, :]) +
        region_norm(all_sel_weights["CR_btag_2"][bkg_idx, :]) +
        region_norm(all_sel_weights["VR_btag_1"][bkg_idx, :]) +
        region_norm(all_sel_weights["CR_btag_1"][bkg_idx, :])
    )

    if "bce" in config.objective:
        loss_value = tomatos.train_utils.bce(ones=sig, zeros=bkg, ones_weights=sig_weights_bce, zeros_weights=bkg_weights_bce)

    cls_log = jnp.nan
    bce_log = jnp.nan
    discovery_log = jnp.nan

    if config.objective == "cls_nn" or config.objective == "cls_var":
        bce_loss = tomatos.train_utils.bce(
            ones=sig,
            zeros=bkg,
            ones_weights=sig_weights_bce,
            zeros_weights=bkg_weights_bce,
        )
        bce_log = bce_loss
        if config.objective == "cls_nn" and step < config.bce_warmup_steps:
            loss_value = bce_loss
        else:
            model, hists = tomatos.workspace.pyhf_model(hists, config, validate_only=validate_only)

            print(model.config.channels)
            assert len(model.config.channels) == len(set(model.config.channels)), "Duplicate channel!"
            cls = neos.loss_from_model(model, loss="cls")
            discovery = neos.loss_from_model(model, loss="discovery")
            cls_log = cls
            discovery_log = discovery
            cls_is_nan = jnp.isnan(cls)
            cls_loss = cls

            if config.objective == "cls_var":
                loss_value = jnp.where(cls_is_nan, bce_loss, cls_loss)
            else:
                # gradually shift from BCE to CLS over cls_anneal_steps to avoid
                # abrupt collapse of the histogram at the transition point
                cls_fraction = jnp.minimum(
                    1.0,
                    (step - config.bce_warmup_steps) / config.cls_anneal_steps,
                )
                annealed = (1.0 - cls_fraction) * bce_loss + cls_fraction * cls_loss
                loss_value = jnp.where(cls_is_nan, bce_loss, annealed)
                # additive BCE regularization: keeps constant separation pressure
                # so the NN can't drift into degenerate solutions when CLS
                # gradient is weak (small S/B)
                loss_value = loss_value + config.bce_reg_weight * bce_loss

    if not validate_only:
            loss_value = tomatos.constraints.penalize_loss(loss_value, hists, config)

    # flatten and reduces to the configured filter
    hists = tomatos.utils.filter_hists(config, hists) if filter_return_hists else hists
    hists["_cls"] = cls_log
    hists["_bce"] = bce_log
    hists["_discovery"] = discovery_log
    return loss_value, hists