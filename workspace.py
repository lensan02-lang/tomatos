import pprint

import jax
import jax.numpy as jnp
import numpy as np
import pyhf
import matplotlib.pyplot as plt


def get_generator_weight_envelope(hists):
    # adapt to the way you've set this up
    nominal = hists["blah"]
    gens = [
        "GEN_MUR05_MUF05_PDF260000",
        "GEN_MUR05_MUF10_PDF260000",
        "GEN_MUR10_MUF05_PDF260000",
        "GEN_MUR10_MUF10_PDF260000",
        "GEN_MUR10_MUF20_PDF260000",
        "GEN_MUR20_MUF10_PDF260000",
        "GEN_MUR20_MUF20_PDF260000",
    ]

    gen_hists = jnp.array([hists[gen] for gen in gens])
    diffs = jnp.abs(gen_hists - nominal)
    max_diffs = jnp.max(diffs, axis=0)
    envelope_up = jnp.array(nominal + max_diffs)
    envelope_down = jnp.array(nominal - max_diffs)
    return envelope_up, envelope_down


def symmetric_up_down_sf(nom, sys):
    relative = jnp.abs((nom - sys) / nom)
    up = 1 + relative
    down = 1 - relative
    # limit to some extent
    up = jnp.where(up > 100, 100, up)
    down = jnp.where(down < 0, 0, down)

    return up, down


def zero_protect(hists, thresh=0.001):
    # opt and fit do not like zeros/tiny numbers
    # go recursively through all hists and replace values with thresh if below
    if isinstance(hists, dict):
        return {key: zero_protect(value, thresh) for key, value in hists.items()}

    if isinstance(hists, jnp.ndarray):
        return jnp.where(hists < thresh, thresh, hists)


def sum_added_back(hists, region, samples, nominal):
    nosys = 0.0
    stat_var = 0.0
    for sample in samples:
        nom = hists[region][sample][nominal]
        sigma = hists[region][sample]["STAT_1UP"] - nom
        nosys = nosys + nom
        stat_var = stat_var + jnp.square(sigma)
    return nosys, jnp.sqrt(stat_var)


def hist_transforms(hists, config, validate_only, update_bkg_shape=False):

    # protect for e.g. divisions in the following
    hists = zero_protect(hists)

    hists["SR_btag_2"]["bkg_estimate"] = {}
    hists["CR_btag_2"]["bkg_estimate"] = {}

    raw_SR_NOSYS = hists["SR_btag_2"]["bkg_estimate_raw"]["NOSYS"]
    raw_SR_STAT_1UP = hists["SR_btag_2"]["bkg_estimate_raw"]["STAT_1UP"]
    raw_SR_STAT_1DOWN = hists["SR_btag_2"]["bkg_estimate_raw"]["STAT_1DOWN"]
    raw_CR_NOSYS = hists["CR_btag_2"]["bkg_estimate_raw"]["NOSYS"]
    raw_CR_STAT_1UP = hists["CR_btag_2"]["bkg_estimate_raw"]["STAT_1UP"]
    raw_CR_STAT_1DOWN = hists["CR_btag_2"]["bkg_estimate_raw"]["STAT_1DOWN"]

    # add back the direct MC prediction of every sample removed from the
    # 1-tag/CR data above (e.g. ttbar) - see sum_added_back(). its own MC
    # stat error enters the STAT band combined in quadrature with the data
    # stat error already in there, since the two are independent sources.
    added_SR_NOSYS, added_SR_STAT_sigma = sum_added_back(
        hists, "SR_btag_2", config.subtract_from_data, config.nominal
    )
    added_CR_NOSYS, added_CR_STAT_sigma = sum_added_back(
        hists, "CR_btag_2", config.subtract_from_data, config.nominal
    )

    hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] = raw_SR_NOSYS + added_SR_NOSYS
    hists["SR_btag_2"]["bkg_estimate"]["STAT_1UP"] = hists["SR_btag_2"]["bkg_estimate"][
        "NOSYS"
    ] + jnp.sqrt(
        jnp.square(raw_SR_STAT_1UP - raw_SR_NOSYS) + jnp.square(added_SR_STAT_sigma)
    )
    hists["SR_btag_2"]["bkg_estimate"]["STAT_1DOWN"] = hists["SR_btag_2"]["bkg_estimate"][
        "NOSYS"
    ] - jnp.sqrt(
        jnp.square(raw_SR_NOSYS - raw_SR_STAT_1DOWN) + jnp.square(added_SR_STAT_sigma)
    )

    hists["CR_btag_2"]["bkg_estimate"]["NOSYS"] = raw_CR_NOSYS + added_CR_NOSYS
    hists["CR_btag_2"]["bkg_estimate"]["STAT_1UP"] = hists["CR_btag_2"]["bkg_estimate"][
        "NOSYS"
    ] + jnp.sqrt(
        jnp.square(raw_CR_STAT_1UP - raw_CR_NOSYS) + jnp.square(added_CR_STAT_sigma)
    )
    hists["CR_btag_2"]["bkg_estimate"]["STAT_1DOWN"] = hists["CR_btag_2"]["bkg_estimate"][
        "NOSYS"
    ] - jnp.sqrt(
        jnp.square(raw_CR_NOSYS - raw_CR_STAT_1DOWN) + jnp.square(added_CR_STAT_sigma)
    )
    cr_pred = hists["CR_btag_2"]["bkg_estimate"]["NOSYS"]
    cr_data = hists["CR_btag_2"]["data"]["NOSYS"]
    cr_pred_stat_up = hists["CR_btag_2"]["bkg_estimate"]["STAT_1UP"]
    cr_data_stat_up = hists["CR_btag_2"]["data"]["STAT_1UP"]

    abs_dev = cr_data - cr_pred

    # statistical uncertainty of abs_dev, from the (independent) Poisson/
    # weighted-stat errors on cr_data and cr_pred
    sigma_cr_data = cr_data_stat_up - cr_data
    sigma_cr_pred = cr_pred_stat_up - cr_pred
    sigma_abs_dev = jnp.sqrt(sigma_cr_data**2 + sigma_cr_pred**2)

    
    safe_cr_pred = cr_pred > 0.02
    rel_dev = jnp.where(safe_cr_pred, abs_dev / cr_pred, 0.0)
    sigma_rel_dev = jnp.where(safe_cr_pred, sigma_abs_dev / cr_pred, 0.0)

    significant_dev = jnp.sqrt(
        jnp.maximum(rel_dev**2 - sigma_rel_dev**2, 0.0)
    )
    # block gradient through the size of this systematic - it still enters
    # the CLs value normally (train and eval see the same loss), but the
    # fit can't reduce it by reshaping cuts/NN output to make CR appear
    # to close better; only the underlying NOSYS (still fully
    # differentiable) can be improved
    significant_dev = jax.lax.stop_gradient(significant_dev)
    config._bkg_shape_dev = significant_dev

    nosys = hists["SR_btag_2"]["bkg_estimate"]["NOSYS"]
    cached = getattr(config, "_bkg_shape_dev", None)
    significant_dev = (
        cached if cached is not None and cached.shape == nosys.shape
        else jnp.zeros_like(nosys)
    )

    # symmetric: widen both sides by the same relative significant
    # deviation, regardless of which direction the CR non-closure points
    # in. Multiplicative, applied to each region's own NOSYS 
    shape_up = 1.0 + significant_dev
    shape_down = jnp.maximum(1.0 - significant_dev, 0.0)

    hists["SR_btag_2"]["bkg_estimate"]["BKG_SHAPE_1UP"] = (
        hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] * shape_up
    )
    hists["SR_btag_2"]["bkg_estimate"]["BKG_SHAPE_1DOWN"] = (
        hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] * shape_down
    )

    # apply to CR (same relative significant_dev, so the constraint from CR
    # actually feeds back consistently)
    hists["CR_btag_2"]["bkg_estimate"]["BKG_SHAPE_1UP"] = (
        hists["CR_btag_2"]["bkg_estimate"]["NOSYS"] * shape_up
    )
    hists["CR_btag_2"]["bkg_estimate"]["BKG_SHAPE_1DOWN"] = (
        hists["CR_btag_2"]["bkg_estimate"]["NOSYS"] * shape_down
    )

    # e.g. if generator weights are available
    # hists["gen_up"], hists["gen_down"] = get_generator_weight_envelope(hists)

    # make sure again after transforms!
    hists = zero_protect(hists)

    return hists



def get_modifiers(hists, region, config, validate_only=False):
    modifiers = {k: [] for k in hists[region]}
    # samples whose STAT uncertainty gets one nuisance parameter per bin
    # instead of a single bin-correlated shift, so the fit can correct
    # bin-by-bin mismatches (e.g. under-predicting one bin and
    # over-predicting the next) instead of only being able to shift the
    # whole histogram up or down together
    per_bin_stat_samples = ["bkg_estimate"]

    for sample in hists[region]:
        for sys in hists[region][sample]:
            if "1UP" in sys:
                if "MY_SF_UNC" in sys:
                    continue
                sys = sys.replace("_1UP", "")
                if sys == "STAT" and sample in per_bin_stat_samples:
                    continue  # handled per-bin below instead
                mod_name = f"{sys}_{sample}" if sys == "STAT" else sys
                modifiers[sample] += (
                    {
                        "name": mod_name,
                        "type": "histosys",
                        "data": {
                            "hi_data": hists[region][sample][sys + "_1UP"],
                            "lo_data": hists[region][sample][sys + "_1DOWN"],
                        },
                    },
                )

    
    for sample in per_bin_stat_samples:
        if sample not in hists[region]:
            continue
        nom = hists[region][sample][config.nominal]
        stat_up = hists[region][sample]["STAT_1UP"]
        stat_down = hists[region][sample]["STAT_1DOWN"]
        for i in range(len(config.bins) - 1):
            hi = jnp.copy(nom).at[i].set(stat_up[i])
            lo = jnp.copy(nom).at[i].set(stat_down[i])
            modifiers[sample] += (
                {
                    "name": f"STAT_{sample}_bin{i + 1}",
                    "type": "histosys",
                    "data": {
                        "hi_data": hi,
                        "lo_data": lo,
                    },
                },
            )

    return modifiers


def sample_spec_from_modifiers(hists, region, config, modifiers, samples):
    return [
        {
            "name": sample,
            "data": hists[region][sample][config.nominal],
            "modifiers": modifiers[sample],
        }
        for sample in samples
    ]


def pyhf_model(hists, config, validate_only=False):
    modifiers_SR = get_modifiers(hists, config.fit_region, config, validate_only)
    sample_spec_SR = sample_spec_from_modifiers(
        hists, config.fit_region, config, modifiers_SR, samples=["bkg_estimate"]
    )

    spec = {
    "channels": [
        {
            "name": config.fit_region,  # SR_btag_2
            "samples": [
                {
                    "name": config.signal_sample,
                    "data": hists[config.fit_region][config.signal_sample][config.nominal],
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None},
                        {
                            "name": "MY_SF_UNC",
                            "type": "histosys",
                            "data": {
                                "hi_data": hists["SR_btag_2"]["ggZH125_vvbb"]["MY_SF_UNC_1UP"],
                                "lo_data": hists["SR_btag_2"]["ggZH125_vvbb"]["MY_SF_UNC_1DOWN"],
                            },
                        },
                        *modifiers_SR[config.signal_sample],
                    ],
                },
                *sample_spec_SR,
            ],
        },
    ],
}

    #pprint.pprint(spec)

    return pyhf.Model(spec, validate=False), hists