import pprint

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


def get_abcd_weight(A, B):
    w_CR = A / B
    errA = jnp.sqrt(A)
    errB = jnp.sqrt(B)
    stat_err_w_CR = w_CR * jnp.sqrt(jnp.square(errA / A) + jnp.square(errB / B))
    return w_CR, stat_err_w_CR


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


def hist_transforms(hists, validate_only):

    # protect for e.g. divisions in the following
    hists = zero_protect(hists)
    # aim to scale btag_1 to btag_2 in SR from ratio in CR
    w_CR, stat_err_w_CR = get_abcd_weight(
        A=jnp.sum(hists["CR_btag_2"]["bkg"]["NOSYS"]),
        B=jnp.sum(hists["CR_btag_1"]["bkg"]["NOSYS"]),
    )
    """print("CR2/CR1:", hists["CR_btag_2"]["bkg"]["NOSYS"], "/", hists["CR_btag_1"]["bkg"]["NOSYS"])
    print("SR1:", hists["SR_btag_1"]["bkg"]["NOSYS"])
    plt.hist2d(hists["CR_btag_2"]["bkg"]["NOSYS"], hists["CR_btag_1"]["bkg"]["NOSYS"], bins=50)
    plt.xlabel("CR_btag_2")
    plt.ylabel("CR_btag_1")
    plt.title("CR Ratio")
    plt.yscale("log")
    plt.xscale
    plt.colorbar()
    plt.show()"""
    #print("ABCD weight (CR2/CR1):", w_CR, "+-", stat_err_w_CR)

    hists["SR_btag_2"]["bkg_estimate"] = {}
    # scale single tagged for bkg estimate
    hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] = (
        hists["SR_btag_1"]["bkg"]["NOSYS"] * w_CR
        
    )
    # current hack until proper diffable norm/stat modifier
    hists["SR_btag_2"]["bkg_estimate"]["STAT_1UP"] = (
        hists["SR_btag_1"]["bkg"]["STAT_1UP"] * w_CR
    )
    hists["SR_btag_2"]["bkg_estimate"]["STAT_1DOWN"] = (
        hists["SR_btag_1"]["bkg"]["STAT_1DOWN"] * w_CR
    )

    # norm uncertainty background estimate
    w_CR_stat_up, w_CR_stat_down = symmetric_up_down_sf(w_CR, w_CR + stat_err_w_CR)
    hists["SR_btag_2"]["bkg_estimate"]["NORM_1UP"] = (
        hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] * w_CR_stat_up
    )
    hists["SR_btag_2"]["bkg_estimate"]["NORM_1DOWN"] = (
        hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] * w_CR_stat_down
    )

    # i leave this as its illustrative to what you could do
    # Backround Shape Uncertainty
    # don't use this when optimizing --> sculpting
    if validate_only:
        hists["VR_btag_2"]["bkg_estimate"] = {}

        hists["VR_btag_2"]["bkg_estimate"]["NOSYS"] = (
            hists["VR_btag_1"]["bkg"]["NOSYS"] * w_CR
        )
        bkg_shapesys_up, bkg_shapesys_down = symmetric_up_down_sf(
            hists["VR_btag_2"]["bkg_estimate"]["NOSYS"],
            hists["VR_btag_2"]["bkg"]["NOSYS"],
        )
        hists["SR_btag_2"]["bkg_estimate"]["BKG_SHAPE_1UP"] = (
            hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] * bkg_shapesys_up
        )
        hists["SR_btag_2"]["bkg_estimate"]["BKG_SHAPE_1DOWN"] = (
            hists["SR_btag_2"]["bkg_estimate"]["NOSYS"] * bkg_shapesys_down
        )
    # e.g. if generator weights are available
    # hists["gen_up"], hists["gen_down"] = get_generator_weight_envelope(hists)

    # make sure again after transforms!
    hists = zero_protect(hists)

    return hists


def get_modifiers(hists, config, validate_only=False):
    modifiers = {k: [] for k in hists[config.fit_region]}
    # autocollect 1UP 1DOWN
    for sample in hists[config.fit_region]:
        for sys in hists[config.fit_region][sample]:
            if "1UP" in sys:
                if "MY_SF_UNC" in sys:
                    continue
                sys = sys.replace("_1UP", "")
                modifiers[sample] += (
                    {
                        "name": sys,
                        "type": "histosys",
                        "data": {
                            "hi_data": hists[config.fit_region][sample][sys + "_1UP"],
                            "lo_data": hists[config.fit_region][sample][sys + "_1DOWN"],
                        },
                    },
                )

    # this an ad-hoc hack for stat uncertainty, and nothing more than
    # that until we have the diffable modifier. Blows up the number of fit
    # parameters unnecessarily for stat unc, which instead of having one
    # parameter per bin, histosy makes a parameter for each bin per
    # modifier
    for sample in [*config.samples]:
        if sample == "bkg":
            continue
        if sample == config.signal_sample:
            continue
        for i in range(len(config.bins) - 1):
            nom = hists[config.fit_region][sample][config.nominal]
            nom_up = jnp.copy(nom)
            stat_up_i = nom_up.at[i].set(
                hists[config.fit_region][sample]["STAT_1UP"][i]
            )
            nom_down = jnp.copy(nom)
            stat_down_i = nom_down.at[i].set(
                hists[config.fit_region][sample]["STAT_1DOWN"][i]
            )
            modifiers[sample] += (
                {
                    "name": f"STAT_{i+1}",
                    "type": "histosys",
                    "data": {
                        "hi_data": jnp.copy(stat_up_i),
                        "lo_data": jnp.copy(stat_down_i),
                    },
                },
            )

    return modifiers


def sample_spec_from_modifiers(hists, config, modifiers, samples):
    # this list is empty here since these are the only two samples, but  auto
    # sets up the modifiers for all up down uncertainties
    sample_spec = [
        {
            "name": sample,
            "data": hists[config.fit_region][sample][config.nominal],
            "modifiers": modifiers[sample],
        }
        for sample in samples
    ]

    return sample_spec


def pyhf_model(hists, config, validate_only=False):
    # here you builds the json worspace structure with hists

    # standard uncerainty modifiers per sample
    # enforce 1UP, 1DOWN for autosetup here
    modifiers = get_modifiers(hists, config, validate_only=validate_only)

    sample_spec = sample_spec_from_modifiers(
        hists, config, modifiers, samples=["bkg"]
    )
    # this is the workspace jsons
    spec = {
        "channels": [
            {
                "name": config.fit_region,
                "samples": [
                    # signal sample
                    {
                        "name": config.signal_sample,
                        "data": hists[config.fit_region][config.signal_sample][
                            config.nominal
                        ],
                        "modifiers": [
                            # signal strength modifier (parameter of interest)
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            },
                            # My custom scale factor uncertainty
                            {
                                "name": "MY_SF_UNC",
                                "type": "histosys",
                                "data": {
                                    "hi_data": hists["SR_btag_2"]["ggZH125_vvbb"][
                                        "MY_SF_UNC_1UP"
                                    ],
                                    "lo_data": hists["SR_btag_2"]["ggZH125_vvbb"][
                                        "MY_SF_UNC_1DOWN"
                                    ],
                                },
                            },
                            *modifiers[config.signal_sample],
                        ],
                    },
                    *sample_spec,
                ],
            }
        ],
    }

    # # this is very handy for debugging when you turn off jit in the main.py
    # if config.debug:
    #pprint.pprint(spec)

    return pyhf.Model(spec, validate=False), hists