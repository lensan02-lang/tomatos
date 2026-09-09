import jax.numpy as jnp
import numpy as np


def min_events_per_bin(h, thresh, intensity=0.001):
    # check where the hist is smaller than thresh and use the relative
    # deviation per bin to penalize
    bin_penalty = jnp.where(h < thresh, (thresh - h) / h, 0)
    penalty = jnp.sum(bin_penalty) * intensity
    return penalty


def entropy_penalty(h, intensity):
    # penalize low entropy (= histogram collapse to one bin)
    # max entropy = log(n_bins), min entropy = 0 (all in one bin)
    p = h / (jnp.sum(h) + 1e-8)
    entropy = -jnp.sum(p * jnp.log(p + 1e-8))
    n_bins = h.shape[0]
    max_entropy = jnp.log(n_bins)
    return intensity * (max_entropy - entropy)


def penalize_loss(loss_value, hists, config):
    loss_value += min_events_per_bin(
        hists["SR_btag_2"]["bkg_estimate"]["NOSYS"],
        thresh=10,
        intensity=0.001,
    )
    # penalize signal and bkg collapsing into one bin
    loss_value += entropy_penalty(
        hists["SR_btag_2"]["bkg_estimate"]["NOSYS"],
        intensity=config.entropy_penalty_weight,
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