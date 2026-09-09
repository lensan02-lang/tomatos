from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import relaxed

import tomatos.select
import tomatos.utils


# modified from relaxed and added weights, its nice to have it here to see
# whats going on
@partial(jax.jit, static_argnames=["density", "reflect_infinities"])
def hist(
    data: jnp.array,
    weights: jnp.array,
    bins: jnp.array,
    bandwidth: float,  # | None = None,
    density: bool = False,
    reflect_infinities: bool = False,
) -> jnp.array:
    """Differentiable histogram, defined via a binned kernel density estimate (bKDE).

    Parameters
    ----------
    data : Array
        1D array of data to histogram.
    weights : Array
        weights to data
    bins : Array
        1D array of bin edges.
    bandwidth : float
        The bandwidth of the kernel. Bigger == lower gradient variance, but more bias.
    density : bool
        Normalise the histogram to unit area.
    reflect_infinities : bool
        If True, define bins at +/- infinity, and reflect their mass into the edge bins.

    Returns
    -------
    Array
        1D array of bKDE counts.
    """

    # 7.2.3 nathan thesis
    # get cumulative counts (area under kde) for each set of bin edges

    # bins=np.array([0,1,2,3])
    # bins.reshape(-1, 1)
    # array([[0],
    #        [1],
    #        [2],
    #        [3]])
    cdf = jsp.stats.norm.cdf(bins.reshape(-1, 1), loc=data, scale=bandwidth)
    # multiply with weight
    cdf = cdf * weights

    # sum kde contributions in each bin
    counts = (cdf[1:, :] - cdf[:-1, :]).sum(axis=1)

    if density:  # normalize by bin width and counts for total area = 1
        db = jnp.array(jnp.diff(bins), float)  # bin spacing
        counts = counts / db / counts.sum(axis=0)

    if reflect_infinities:
        counts = (
            counts[1:-1]
            + jnp.array([counts[0]] + [0] * (len(counts) - 3))
            + jnp.array([0] * (len(counts) - 3) + [counts[-1]])
        )

    return counts


@partial(jax.jit, static_argnames=["nn_arch", "nn_inputs_idx_end"])
def get_nn_output(pars, data, nn_arch, nn_inputs_idx_end, temperature=1.0):
    """Inference mode — dropout disabled."""
    nn = eqx.combine(pars["nn"], nn_arch)

    def predict_sample(i):
        sample_data = data[i, :, :nn_inputs_idx_end]
        return jax.vmap(lambda x: nn(x, inference=True))(sample_data).ravel() / temperature

    return jax.vmap(predict_sample)(jnp.arange(data.shape[0]))


def get_nn_output_training(pars, data, nn_arch, nn_inputs_idx_end, temperature=1.0):
    """Training mode — dropout active with random keys per event."""
    import numpy as np
    nn = eqx.combine(pars["nn"], nn_arch)
    base_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
    results = []
    for i in range(data.shape[0]):
        sample_data = data[i, :, :nn_inputs_idx_end]
        n_events = sample_data.shape[0]
        sample_key = jax.random.fold_in(base_key, i)
        event_keys = jax.random.split(sample_key, n_events)
        out = jax.vmap(lambda x, k: nn(x, key=k, inference=False))(sample_data, event_keys)
        results.append(out.ravel() / temperature)
    return jnp.stack(results)


# jitting however not of much help here
@partial(jax.jit, static_argnames=["objective", "cls_var_idx", "w2"])
def compute_hist_wrapper(
    i,
    objective,
    data,
    cls_var_idx,
    nn_output,
    weights,
    scale,
    bw,
    bins,
    w2=False,
):
    # lots of args due to the pure function paradigm
    # these ifs only work because of static_argnames
    if objective == "cls_var":
        sample_data = data[i, :, cls_var_idx]
    elif objective in ["cls_nn","bce"]:
        sample_data = nn_output[i, :]

    sample_weights = weights[i, :]

    if w2:
        sample_weights = jnp.power(sample_weights, 2)

    # Scale works also as an estimate for w2
    return (
        hist(
            data=sample_data,
            weights=sample_weights,
            bandwidth=bw,
            bins=bins,
        )
        * scale[i]
    )


def fill_hists(
    pars,
    data,
    config,
    sel_weights,
    scale,
    validate_only,
    step=0,
):
    # any magic in here will at the end just call the upper hist() function

    bins = jnp.array([0, *pars["bins"], 1]) if config.include_bins else config.bins

    # make hists sharp if validation
    bw = 1e-20 if validate_only else pars["bw"]

    # this will hold: hists[sel][sample][sys]
    hists = {sel: {sample: {} for sample in config.samples} for sel in sel_weights}

    # get nn output
    if config.objective in ["cls_nn","bce"]:
        nn_output = get_nn_output(
            pars,
            data,
            config.nn_arch,
            config.nn_inputs_idx_end,
        )
    else:
        nn_output = None

    compute_hist = partial(
        compute_hist_wrapper,
        objective=config.objective,
        data=data,
        cls_var_idx=config.cls_var_idx,
        nn_output=nn_output,
        scale=scale,
        bw=bw,
        bins=bins,
    )
    # calc all hists for all samples in fit region
    hists_vector = jax.vmap(
        lambda i: compute_hist(i, weights=sel_weights[config.fit_region])
    )(jnp.arange(len(config.sample_sys)))

    # this is the sequential version
    # hists_vector = []
    # for i in range(len(config.sample_sys)):
    #     hist = compute_hist(i, weights=sel_weights[config.fit_region])
    #     hists_vector.append(hist)

    for sample_sys, h in zip(config.sample_sys, hists_vector):
        sample, sys = config.sample_sys_dict[sample_sys]
        hists[config.fit_region][sample][sys] = h

    def extra_hists(hists):

        # Compute w2 histograms only for the ones we need
        # going over len(config.samples) works because NOSYS are the first ones per
        # sample, see config
        hists_nominal_w2_vector = jax.vmap(
            lambda i: compute_hist(
                i,
                weights=sel_weights["SR_btag_2"],
                w2=True,
            )
        )(jnp.arange(len(config.samples)))

        # workaround stat up and down hists
        for sample, h_w2 in zip(config.samples, hists_nominal_w2_vector):
            sigma = jnp.sqrt(h_w2)
            hists["SR_btag_2"][sample]["STAT_1UP"] = (
                hists["SR_btag_2"][sample][config.nominal] + sigma
            )
            hists["SR_btag_2"][sample]["STAT_1DOWN"] = (
                hists["SR_btag_2"][sample][config.nominal] - sigma
            )

        signal_idx = config.sample_sys.index("ggZH125_vvbb_NOSYS")
        bkg_idx = config.sample_sys.index("bkg_NOSYS")
        data_idx = config.sample_sys.index("data_NOSYS")

        # MC bkg truth in the ABCD regions, used only for closure plots
        # (log_abcd_closure, log_abcd_vr_closure, log_sr_comparison) -
        # kept separate from the data-driven estimate below since it's
        # the ground truth the estimate is checked against
        hists["SR_btag_1"]["bkg"]["NOSYS"] = compute_hist(
            i=bkg_idx, weights=sel_weights["SR_btag_1"]
        )
        h_w2_SR_btag_1 = compute_hist(
            i=bkg_idx, weights=sel_weights["SR_btag_1"], w2=True
        )
        sigma = jnp.sqrt(h_w2_SR_btag_1)
        hists["SR_btag_1"]["bkg"]["STAT_1UP"] = hists["SR_btag_1"]["bkg"]["NOSYS"] + sigma
        hists["SR_btag_1"]["bkg"]["STAT_1DOWN"] = hists["SR_btag_1"]["bkg"]["NOSYS"] - sigma

        hists["CR_btag_2"]["bkg"]["NOSYS"] = compute_hist(
            i=bkg_idx, weights=sel_weights["CR_btag_2"]
        )
        hists["CR_btag_1"]["bkg"]["NOSYS"] = compute_hist(
            i=bkg_idx, weights=sel_weights["CR_btag_1"]
        )

        hists["VR_btag_2"]["bkg"]["NOSYS"] = compute_hist(
            i=bkg_idx, weights=sel_weights["VR_btag_2"]
        )
        hists["VR_btag_1"]["bkg"]["NOSYS"] = compute_hist(
            i=bkg_idx, weights=sel_weights["VR_btag_1"]
        )

        # gebraucht von workspace.hist_transforms (CR-closure -> BKG_SHAPE)
        # und train_utils.log_matrix_cr_closure
        hists["CR_btag_2"]["data"]["NOSYS"] = compute_hist(
            i=data_idx, weights=sel_weights["CR_btag_2"]
        )
        h_w2_CR_btag_2_data = compute_hist(
            i=data_idx, weights=sel_weights["CR_btag_2"], w2=True
        )
        sigma_data_cr2 = jnp.sqrt(h_w2_CR_btag_2_data)
        hists["CR_btag_2"]["data"]["STAT_1UP"] = (
            hists["CR_btag_2"]["data"]["NOSYS"] + sigma_data_cr2
        )
        hists["CR_btag_2"]["data"]["STAT_1DOWN"] = (
            hists["CR_btag_2"]["data"]["NOSYS"] - sigma_data_cr2
        )

        hists["VR_btag_2"]["data"]["NOSYS"] = compute_hist(
            i=data_idx, weights=sel_weights["VR_btag_2"]
        )
        h_w2_VR_btag_2_data = compute_hist(
            i=data_idx, weights=sel_weights["VR_btag_2"], w2=True
        )
        sigma_data_vr2 = jnp.sqrt(h_w2_VR_btag_2_data)
        hists["VR_btag_2"]["data"]["STAT_1UP"] = (
            hists["VR_btag_2"]["data"]["NOSYS"] + sigma_data_vr2
        )
        hists["VR_btag_2"]["data"]["STAT_1DOWN"] = (
            hists["VR_btag_2"]["data"]["NOSYS"] - sigma_data_vr2
        )

        hists["VR_btag_1"]["data"]["NOSYS"] = compute_hist(
            i=data_idx, weights=sel_weights["VR_btag_1"]
        )
        h_w2_VR_btag_1_data = compute_hist(
            i=data_idx, weights=sel_weights["VR_btag_1"], w2=True
        )
        sigma_data_vr1 = jnp.sqrt(h_w2_VR_btag_1_data)
        hists["VR_btag_1"]["data"]["STAT_1UP"] = (
            hists["VR_btag_1"]["data"]["NOSYS"] + sigma_data_vr1
        )
        hists["VR_btag_1"]["data"]["STAT_1DOWN"] = (
            hists["VR_btag_1"]["data"]["NOSYS"] - sigma_data_vr1
        )

        hists["VR_btag_0"]["data"]["NOSYS"] =compute_hist(
            i=data_idx, weights=sel_weights["VR_btag_0"]
        )
        hists["CR_btag_0"]["data"]["NOSYS"] =compute_hist(
                    i=data_idx, weights=sel_weights["CR_btag_0"]
                )
        hists["SR_btag_0"]["data"]["NOSYS"] =compute_hist(
                    i=data_idx, weights=sel_weights["SR_btag_0"]
                )
        hists["VR_btag_0"]["data"]["NOSYS"] =compute_hist(
            i=data_idx, weights=sel_weights["VR_btag_0"]
        )
        hists["CR_btag_0"]["data"]["NOSYS"] =compute_hist(
                    i=data_idx, weights=sel_weights["CR_btag_0"]
                )
        hists["SR_btag_0"]["data"]["NOSYS"] =compute_hist(
                    i=data_idx, weights=sel_weights["SR_btag_0"]
                )

        # Matrix-Methode: roher (noch ohne added-back ttbar) bkg_estimate,
        # direkt aus den pro-Event-Gewichten (select.matrix_method_bkg_weight)
        # histogrammiert. "raw" weil workspace.py die
        # subtract_from_data-Samples (z.B. ttbar) noch direkt addiert.
        # Ihre faelschliche "fake"-Kontamination in den eigenen TL/LT/LL-
        # Events wird weiter unten schon rausgerechnet (bevor die
        # subtract_from_data-Schleife laeuft), damit sie hier nicht
        # zusaetzlich zum add-back doppelt gezaehlt wird
        hists["SR_btag_2"]["bkg_estimate_raw"] = {}
        hists["SR_btag_2"]["bkg_estimate_raw"]["NOSYS"] = compute_hist(
            i=data_idx, weights=sel_weights["SR_bkg_estimate"]
        )
        h_w2_SR_bkg = compute_hist(
            i=data_idx, weights=sel_weights["SR_bkg_estimate"], w2=True
        )
        sigma_SR_bkg = jnp.sqrt(h_w2_SR_bkg)
        hists["SR_btag_2"]["bkg_estimate_raw"]["STAT_1UP"] = (
            hists["SR_btag_2"]["bkg_estimate_raw"]["NOSYS"] + sigma_SR_bkg
        )
        hists["SR_btag_2"]["bkg_estimate_raw"]["STAT_1DOWN"] = (
            hists["SR_btag_2"]["bkg_estimate_raw"]["NOSYS"] - sigma_SR_bkg
        )

        hists["CR_btag_2"]["bkg_estimate_raw"] = {}
        hists["CR_btag_2"]["bkg_estimate_raw"]["NOSYS"] = compute_hist(
            i=data_idx, weights=sel_weights["CR_bkg_estimate"]
        )
        h_w2_CR_bkg = compute_hist(
            i=data_idx, weights=sel_weights["CR_bkg_estimate"], w2=True
        )
        sigma_CR_bkg = jnp.sqrt(h_w2_CR_bkg)
        hists["CR_btag_2"]["bkg_estimate_raw"]["STAT_1UP"] = (
            hists["CR_btag_2"]["bkg_estimate_raw"]["NOSYS"] + sigma_CR_bkg
        )
        hists["CR_btag_2"]["bkg_estimate_raw"]["STAT_1DOWN"] = (
            hists["CR_btag_2"]["bkg_estimate_raw"]["NOSYS"] - sigma_CR_bkg
        )


        # subtract known MC contamination (e.g. ttbar) from the data-driven
        # ABCD template before it becomes bkg_estimate
        for subtract_sample in config.subtract_from_data:
            subtract_idx = config.sample_sys.index(
                f"{subtract_sample}_{config.nominal}"
            )
            for region in [
                "VR_btag_2",
                "VR_btag_1",
            ]:
                h_subtract = compute_hist(i=subtract_idx, weights=sel_weights[region])
                for sys in ["NOSYS", "STAT_1UP", "STAT_1DOWN"]:
                    hists[region]["data"][sys] = hists[region]["data"][sys] - h_subtract

            # Matrix-Methode: dieselbe Kontamination auch aus dem
            # matrix-basierten bkg_estimate_raw rausrechnen, BEVOR
            # workspace.hist_transforms subtract_sample dort wieder direkt
            # addiert. sel_weights["SR_bkg_estimate"]/["CR_bkg_estimate"]
            # sind pro-Event-Gewichte, die fuer JEDE sample_sys-Zeile aus
            # ihren EIGENEN Jet-pt/eta ueber matrix_method_bkg_weight
            # berechnet wurden - fuer subtract_sample (z.B. ttbar) sagt
            # sel_1/sel_2 pro Event schon exakt (nicht probabilistisch),
            # ob/wie es TL/LT/LL/TT tagged, kein er/ef-Nachschlagen noetig,
            # es laeuft nur automatisch durch dieselbe Funktion mit. Ohne
            # das hier wuerde subtract_sample doppelt gezaehlt: einmal
            # (faelschlich als "fake") ueber die
            # matrix_method_bkg_weight-Gewichtung seiner eigenen
            # TL/LT/LL-Events, einmal direkt ueber den add-back unten in
            # workspace.hist_transforms.
            for bkg_key, sr_or_cr in [
                ("SR_bkg_estimate", "SR_btag_2"),
                ("CR_bkg_estimate", "CR_btag_2"),
            ]:
                h_subtract_bkg = compute_hist(
                    i=subtract_idx, weights=sel_weights[bkg_key]
                )
                for sys in ["NOSYS", "STAT_1UP", "STAT_1DOWN"]:
                    hists[sr_or_cr]["bkg_estimate_raw"][sys] = (
                        hists[sr_or_cr]["bkg_estimate_raw"][sys] - h_subtract_bkg
                    )

            # the w_CR-scaled estimate above only ever reinstates the part
            # of the background that transfers from 1-tag to 2-tag like the
            # mistag rate does - since subtract_sample was just removed
            # from the 1-tag/CR data it's built from, its 2-tag
            # contribution has to come back in directly from its own MC in
            # the 2-tag regions instead (added onto bkg_estimate in
            # workspace.hist_transforms). SR_btag_2 is config.fit_region,
            # so NOSYS/STAT_1UP/STAT_1DOWN for subtract_sample are already
            # filled by the vmap block at the top of this function;
            # VR_btag_2 needs the same explicitly since it isn't the fit
            # region.
            hists["CR_btag_2"][subtract_sample]["NOSYS"] = compute_hist(
                i=subtract_idx, weights=sel_weights["CR_btag_2"]
            )
            h_w2_CR_btag_2_subtract = compute_hist(
                i=subtract_idx, weights=sel_weights["CR_btag_2"], w2=True
            )
            sigma_subtract = jnp.sqrt(h_w2_CR_btag_2_subtract)
            hists["CR_btag_2"][subtract_sample]["STAT_1UP"] = (
                hists["CR_btag_2"][subtract_sample]["NOSYS"] + sigma_subtract
            )
            hists["CR_btag_2"][subtract_sample]["STAT_1DOWN"] = (
                hists["CR_btag_2"][subtract_sample]["NOSYS"] - sigma_subtract
            )

        hists["CR_btag_2"][config.signal_sample]["NOSYS"] = compute_hist(
            i=signal_idx, weights=sel_weights["CR_btag_2"]
        )


        # some special signal weight unc, e.g. btag sf
        hists["SR_btag_2"]["ggZH125_vvbb"]["MY_SF_UNC_1UP"] = compute_hist(
            i=signal_idx, weights=sel_weights["SR_btag_2_my_sf_unc_up"]
        )
        hists["SR_btag_2"]["ggZH125_vvbb"]["MY_SF_UNC_1DOWN"] = compute_hist(
            i=signal_idx, weights=sel_weights["SR_btag_2_my_sf_unc_down"]
        )

        return hists

    hists = extra_hists(hists)

    if config.signal_scale != 1.0:
        for sel in hists:
            if config.signal_sample in hists[sel]:
                hists[sel][config.signal_sample] = {
                    sys: h * config.signal_scale
                    for sys, h in hists[sel][config.signal_sample].items()
                }

    return hists