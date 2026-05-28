import jax.numpy as jnp
import numpy as np


def min_events_per_bin(h, thresh, intensity=0.001):
    # check where the hist is smaller than thresh and use the relative
    # deviation per bin to penalize
    bin_penalty = jnp.where(h < thresh, (thresh - h) / h, 0)
    penalty = jnp.sum(bin_penalty) * intensity
    return penalty


def penalize_loss(loss_value, hists):
    loss_value += min_events_per_bin(
        hists["SR_btag_2"]["bkg"]["NOSYS"],
        thresh=10,
        intensity=10.0, 
    )
    loss_value += min_events_per_bin(
        hists["SR_btag_2"]["bkg_estimate"]["NOSYS"],
        thresh=10,
        intensity=10.0,
    )
    return loss_value


def opt_pars(config, opt_pars):
    # a large step can gow below 0 which breaks opt since this flips
    # the cdf (not the pdf) used for histogram calculation, also not go
    # below a minimum to maintain gradients
    opt_pars["bw"] = np.maximum(config.bw_min, np.abs(opt_pars["bw"]))

    if config.include_bins:
        # maintain order and avoid going out of 0, 1 range
        opt_pars["bins"] = np.clip(np.sort(np.abs(opt_pars["bins"])), 1e-6, 1 - 1e-6)
    return opt_pars