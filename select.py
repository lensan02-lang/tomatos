from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import relaxed
import tomatos.utils

import numpy as np


# @partial(jax.jit, static_argnames=["config", "validate_only"])
# @jax.jit
def cuts(pars, data, config, validate_only, step=0):
    cut_weights = jnp.ones(data.shape[:2])
    if config.cuts_start_step is not None:
        active = step >= config.cuts_start_step
    else:
        active = config.include_cuts
    if not active:
        return cut_weights
    # remove approximation with large slope for validation --> sharp cuts
    slope = 1e20 if validate_only else config.slope
    # collect them over all cuts and apply to weights once
    for var, var_dict in config.opt_cuts.items():
        if var_dict["keep"] == "window":
            center = pars["cut_" + var + "_center"]
            half_width = jnp.exp(pars["cut_" + var + "_logwidth"]) / 2
            lo = center - half_width
            hi = center + half_width
            cut_weights *= relaxed.cut(
                data=data[:, :, var_dict["idx"]],
                cut_val=lo,
                slope=slope,
                keep="above",
            )
            cut_weights *= relaxed.cut(
                data=data[:, :, var_dict["idx"]],
                cut_val=hi,
                slope=slope,
                keep="below",
            )
        else:
            cut_weights *= relaxed.cut(
                data=data[:, :, var_dict["idx"]],
                cut_val=pars["cut_" + var],
                slope=slope,
                keep=var_dict["keep"],
            )

    # NB
    # if you wonder why not doing cuts like this:
    # var = jnp.where(var > cut_param, var, 0)
    # discontinuous --> no gradient

    return cut_weights


# @partial(jax.jit, static_argnames=["config"])
# @jax.jit
def _lookup_efficiency(pt, eta, curve):
    """Efficiency lookup for one curve entry.

    Falls back to plain 1D pt-interpolation if the curve has no
    "eta_centers" (old efficiency_curves*.json format). If it does, does a
    2D bilinear lookup in (pt, eta) instead. "efficiency" is then expected
    to have shape (len(pt_centers), len(eta_centers)).
    """
    if "eta_centers" not in curve:
        return jnp.interp(pt, curve["pt_centers"], curve["efficiency"])

    pt_centers = curve["pt_centers"]
    eta_centers = curve["eta_centers"]
    pt_idx = jnp.interp(pt, pt_centers, jnp.arange(pt_centers.shape[0], dtype=pt_centers.dtype))
    eta_idx = jnp.interp(eta, eta_centers, jnp.arange(eta_centers.shape[0], dtype=eta_centers.dtype))
    return map_coordinates(curve["efficiency"], [pt_idx, eta_idx], order=1, mode="nearest")


def matrix_method_bkg_weight(data, config, return_components=False):
    j1_pt_idx = config.vars.index("j1_pt")
    j2_pt_idx = config.vars.index("j2_pt")
    j1_eta_idx = config.vars.index("j1_eta")
    j2_eta_idx = config.vars.index("j2_eta")
    j1_pt_scale = data[:, :, j1_pt_idx]
    j2_pt_scale = data[:, :, j2_pt_idx]
    j1_eta_scale = data[:, :, j1_eta_idx]
    j2_eta_scale = data[:, :, j2_eta_idx]
    j1_tag = data[:, :, config.vars.index("sel_1")]
    j2_tag = data[:, :, config.vars.index("sel_2")]

    #scale back to original pt/eta range for efficiency curve lookup
    j1_pt = tomatos.utils.inverse_min_max_scale(config, j1_pt_scale, j1_pt_idx)
    j2_pt = tomatos.utils.inverse_min_max_scale(config, j2_pt_scale, j2_pt_idx)
    j1_eta = tomatos.utils.inverse_min_max_scale(config, j1_eta_scale, j1_eta_idx)
    j2_eta = tomatos.utils.inverse_min_max_scale(config, j2_eta_scale, j2_eta_idx)

    ec = config.efficiency_curves
    er1 = _lookup_efficiency(j1_pt, j1_eta, ec["er_j1"])
    ef1 = _lookup_efficiency(j1_pt, j1_eta, ec["ef_j1"])
    er2 = _lookup_efficiency(j2_pt, j2_eta, ec["er_j2"])
    ef2 = _lookup_efficiency(j2_pt, j2_eta, ec["ef_j2"])

    denom = (ef1 - er1) * (ef2 - er2)

    w_TL = (ef2 * er1 * er2 * (1 - ef1)) / denom
    w_LT = (ef1 * er1 * er2 * (1 - ef2)) / denom
    w_LL = (-ef1 * ef2 * er1 * er2) / denom

    is_TL = j1_tag * (1 - j2_tag)
    is_LT = (1 - j1_tag) * j2_tag
    is_LL = (1 - j1_tag) * (1 - j2_tag)

    contrib_TL = is_TL * w_TL
    contrib_LT = is_LT * w_LT
    contrib_LL = is_LL * w_LL

    if return_components:
        return contrib_TL, contrib_LT, contrib_LL
    return contrib_TL + contrib_LT + contrib_LL


def events(data, config, base_weights):
    btag_1 = data[:, :, config.vars.index("bool_btag_1")]
    btag_2 = data[:, :, config.vars.index("bool_btag_2")]
    btag_0 = data[:, :, config.vars.index("bool_btag_0")]
    j1_tag = data[:, :, config.vars.index("sel_1")]
    j2_tag = data[:, :, config.vars.index("sel_2")]
    h_m_idx = config.vars.index("Xhh")
    h_m = data[:, :, h_m_idx]

    SR = h_m < 1.6
    VR = (1.6 < h_m) & (h_m < 3.0)
    CR = h_m > 3.0

    # welcher der beiden Jets ist das getaggte, innerhalb der 1-tag-Kategorie
    tag_j1_only = j1_tag * (1 - j2_tag)   # n_TL: j1 getaggt, j2 nicht
    tag_j2_only = (1 - j1_tag) * j2_tag   # n_LT: j2 getaggt, j1 nicht

    # Matrix-Methode: Gewicht pro Event fuer den vorhergesagten Beitrag zur
    # "fake, beide getaggt"-Population - wird direkt aus der jeweiligen
    # Region selbst gebaut (SR aus SR, VR aus VR), nicht aus der CR
    # extrapoliert wie bei ABCD, weil die Effizienzkurven pT- statt
    # Xhh-abhaengig sind
    bkg_weight = matrix_method_bkg_weight(data, config)

    weights = {
        "SR_btag_1": base_weights * SR * btag_1,
        "SR_btag_2": base_weights * SR * btag_2,
        "SR_btag_0": base_weights * SR * btag_0,
        "SR_btag_1_j1": base_weights * SR * tag_j1_only,
        "SR_btag_1_j2": base_weights * SR * tag_j2_only,
        "VR_btag_1": base_weights * VR * btag_1,
        "VR_btag_2": base_weights * VR * btag_2,
        "VR_btag_0": base_weights * VR * btag_0,
        "VR_btag_1_j1": base_weights * VR * tag_j1_only,
        "VR_btag_1_j2": base_weights * VR * tag_j2_only,
        "CR_btag_1": base_weights * CR * btag_1,
        "CR_btag_2": base_weights * CR * btag_2,
        "CR_btag_0": base_weights * CR * btag_0,
        "CR_btag_1_j1": base_weights * CR * tag_j1_only,
        "CR_btag_1_j2": base_weights * CR * tag_j2_only,
        "SR_btag_2_my_sf_unc_up": base_weights * SR * btag_2 * data[:, :, config.vars.index("sf_up")],
        "SR_btag_2_my_sf_unc_down": base_weights * SR * btag_2 * data[:, :, config.vars.index("sf_down")],
        "SR_bkg_estimate": base_weights * SR * bkg_weight,
        "CR_bkg_estimate": base_weights * CR * bkg_weight,
    }
    return weights