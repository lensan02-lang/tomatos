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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
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


def binary_cross_entropy_probs(probs, labels, weights=None, eps=1e-7):
    # NeuralNetwork's last layer is already jax.nn.sigmoid, so nn_output is
    # a probability in [0,1], not a raw logit. optax.sigmoid_binary_cross_entropy
    # expects raw logits and applies its own internal sigmoid - feeding it an
    # already-sigmoided value double-squashes it (effectively
    # sigmoid(sigmoid(raw))), which can never saturate near 0 or 1 and
    # distorts the gradient for both classes. Compute BCE directly on the
    # probability instead.
    probs = jnp.clip(probs, eps, 1 - eps)
    loss = -(labels * jnp.log(probs) + (1 - labels) * jnp.log(1 - probs))
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
    probs = jnp.concatenate((ones, zeros))
    if ones_weights is None and zeros_weights is None:
        return binary_cross_entropy_probs(
            probs,
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
    return binary_cross_entropy_probs(
        probs,
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

def _total_unc(sample_dict, nom_norm, norm_sum):
    """Combine all UP/DOWN systematics in quadrature → (total_up, total_down)."""
    sigma_up_sq = np.zeros_like(nom_norm)
    sigma_down_sq = np.zeros_like(nom_norm)
    for key, h in sample_dict.items():
        if "1UP" not in key:
            continue
        sys_name = key.replace("_1UP", "")
        down_key = sys_name + "_1DOWN"
        if down_key not in sample_dict:
            continue
        up   = np.array(h) / norm_sum
        down = np.array(sample_dict[down_key]) / norm_sum
        sigma_up_sq   += (np.maximum(up - nom_norm, 0)) ** 2 + (np.maximum(down - nom_norm, 0)) ** 2
        sigma_down_sq += (np.maximum(nom_norm - up, 0)) ** 2 + (np.maximum(nom_norm - down, 0)) ** 2
    return nom_norm + np.sqrt(sigma_up_sq), nom_norm - np.sqrt(sigma_down_sq)

def log_matrix_cr_closure(config, opt_pars, data, scale, step_i):
    hists = tomatos.pipeline.make_hists(
        opt_pars, data, config, scale, filter_return_hists=False, step=step_i,
    )
    cr_data = np.array(hists["CR_btag_2"]["data"]["NOSYS"])
    cr_pred = np.array(hists["CR_btag_2"]["bkg_estimate"]["NOSYS"])

    # per-bin Poisson/weighted-stat sigma, derived from STAT_1UP - NOSYS
    # (see histograms.py / workspace.hist_transforms)
    sigma_cr_data = np.array(hists["CR_btag_2"]["data"]["STAT_1UP"]) - cr_data
    sigma_cr_pred = np.array(hists["CR_btag_2"]["bkg_estimate"]["STAT_1UP"]) - cr_pred

    sum_data, sum_pred = cr_data.sum(), cr_pred.sum()

    # region-total sigma: sum of per-bin variances (bins partition the
    # region, so this is exactly the Poisson/weighted error of the summed
    # yield)
    sigma_sum_data = np.sqrt(np.sum(sigma_cr_data**2))
    sigma_sum_pred = np.sqrt(np.sum(sigma_cr_pred**2))

    if sum_data < 1e-10 or sum_pred < 1e-10:
        return None, None, None

    non_closure = (sum_data - sum_pred) / sum_pred
    sigma_non_closure = np.sqrt(
        (sigma_sum_data / sum_pred) ** 2
        + (sum_data * sigma_sum_pred / sum_pred**2) ** 2
    )

    # Normalisierte Shapes
    cr_data_norm = cr_data / sum_data
    cr_pred_norm = cr_pred / sum_pred

    # relative stat uncertainty on the normalised shapes 
    with np.errstate(invalid="ignore", divide="ignore"):
        cr_data_norm_sigma = cr_data_norm * np.sqrt(
            (sigma_cr_data / np.where(cr_data > 0, cr_data, np.inf)) ** 2
            + (sigma_sum_data / sum_data) ** 2
        )
        cr_pred_norm_sigma = cr_pred_norm * np.sqrt(
            (sigma_cr_pred / np.where(cr_pred > 0, cr_pred, np.inf)) ** 2
            + (sigma_sum_pred / sum_pred) ** 2
        )

    # === Bin-weise Non-closure === (absolute, not normalised)
    non_closure_per_bin = np.where(
        cr_pred > 1e-8, (cr_data - cr_pred) / cr_pred, np.nan
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        non_closure_per_bin_sigma = np.where(
            cr_pred > 1e-8,
            np.sqrt(
                (sigma_cr_data / cr_pred) ** 2
                + (cr_data * sigma_cr_pred / cr_pred**2) ** 2
            ),
            np.nan,
        )

    bins = config.bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)

    # === TL/LT/LL-Aufschluesselung ===
    # Die drei Kategorien, aus denen sich bkg_estimate zusammensetzt: TL/LT
    # sind meist klein und positiv, LL ist so gut wie immer die weitaus
    # groesste Population (0-Tag) und ihr Beitrag ist negativ (siehe
    # select.matrix_method_bkg_weight) - bkg_estimate ist der kleine Rest
    # einer fast vollstaendigen Ausloeschung zwischen ihnen. Ohne diesen
    # Breakdown sieht man nur die Netto-Zahl und jagt Symptome statt der
    # eigentlichen Ursache (winzige relative Fehler in er/ef an der von LL
    # dominierten pT werden durch die riesige LL-Population brutal verstaerkt).
    data_idx = config.sample_sys.index(f"data_{config.nominal}")
    h_m = data[:, :, config.vars.index("Xhh")]
    CR = h_m > 3.0
    j1_tag = data[:, :, config.vars.index("sel_1")]
    j2_tag = data[:, :, config.vars.index("sel_2")]

    cut_weights = tomatos.select.cuts(opt_pars, data, config, validate_only=False, step=step_i)
    base_weights = data[:, :, config.weight_idx] * cut_weights

    contrib_TL, contrib_LT, contrib_LL = tomatos.select.matrix_method_bkg_weight(
        data, config, return_components=True
    )

    nn_output = None
    if config.objective in ["cls_nn", "bce"]:
        nn_output = tomatos.histograms.get_nn_output(
            opt_pars, data, config.nn_arch, config.nn_inputs_idx_end
        )
    compute_hist_breakdown = partial(
        tomatos.histograms.compute_hist_wrapper,
        objective=config.objective,
        data=data,
        cls_var_idx=config.cls_var_idx,
        nn_output=nn_output,
        scale=scale,
        bw=opt_pars["bw"],
        bins=bins,
    )
    h_TL = np.array(compute_hist_breakdown(data_idx, weights=base_weights * CR * contrib_TL))
    h_LT = np.array(compute_hist_breakdown(data_idx, weights=base_weights * CR * contrib_LT))
    h_LL = np.array(compute_hist_breakdown(data_idx, weights=base_weights * CR * contrib_LL))
    h_net = h_TL + h_LT + h_LL  # = bkg_estimate_raw, vor dem add-back

    n_TL = int((CR * j1_tag * (1 - j2_tag))[data_idx].sum())
    n_LT = int((CR * (1 - j1_tag) * j2_tag)[data_idx].sum())
    n_LL = int((CR * (1 - j1_tag) * (1 - j2_tag))[data_idx].sum())

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        4, 1, sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 1]},
        figsize=(8, 15),
    )
    LABEL_SIZE  = 14
    LEGEND_SIZE = 13
    TICK_SIZE   = 14

    # === Ax0: CR2 obs vs pred, absolut (nicht normiert) ===
    ax0.stairs(cr_data, bins, color="forestgreen", lw=1.5,
               label="CR2 observed (data)")
    ax0.fill_between(
        bins, np.r_[cr_data - sigma_cr_data, (cr_data - sigma_cr_data)[-1]],
        np.r_[cr_data + sigma_cr_data, (cr_data + sigma_cr_data)[-1]],
        step="post", alpha=0.2, color="forestgreen",
    )
    ax0.stairs(cr_pred, bins, color="orange", lw=1.5,
               linestyle="--", label="CR2 predicted (bkg_estimate)")
    ax0.fill_between(
        bins, np.r_[cr_pred - sigma_cr_pred, (cr_pred - sigma_cr_pred)[-1]],
        np.r_[cr_pred + sigma_cr_pred, (cr_pred + sigma_cr_pred)[-1]],
        step="post", alpha=0.2, color="orange",
    )
    ax0.set_ylabel("Events (absolute)", fontsize=LABEL_SIZE)
    ax0.set_title(f"Matrix-Method CR Closure — step {step_i}\n"
                  f"CR2 obs = {sum_data:.1f} ± {sigma_sum_data:.1f},  "
                  f"CR2 pred = {sum_pred:.1f} ± {sigma_sum_pred:.1f},  "
                  f"global non-closure = {non_closure*100:.1f} ± {sigma_non_closure*100:.1f}%",
                  fontsize=LABEL_SIZE)
    ax0.legend(fontsize=LEGEND_SIZE)
    ax0.tick_params(labelsize=TICK_SIZE)

    # === Ax1: CR2 obs vs pred, normierte Shapes ===
    ax1.stairs(cr_data_norm, bins, color="forestgreen", lw=1.5,
               label="CR2 observed (data)")
    ax1.fill_between(
        bins, np.r_[cr_data_norm - cr_data_norm_sigma, (cr_data_norm - cr_data_norm_sigma)[-1]],
        np.r_[cr_data_norm + cr_data_norm_sigma, (cr_data_norm + cr_data_norm_sigma)[-1]],
        step="post", alpha=0.2, color="forestgreen",
    )
    ax1.stairs(cr_pred_norm, bins, color="orange",      lw=1.5,
               linestyle="--", label="CR2 predicted (bkg_estimate)")
    ax1.fill_between(
        bins, np.r_[cr_pred_norm - cr_pred_norm_sigma, (cr_pred_norm - cr_pred_norm_sigma)[-1]],
        np.r_[cr_pred_norm + cr_pred_norm_sigma, (cr_pred_norm + cr_pred_norm_sigma)[-1]],
        step="post", alpha=0.2, color="orange",
    )
    ax1.set_ylabel("Normalised Events", fontsize=LABEL_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.tick_params(labelsize=TICK_SIZE)

    # === Ax2: TL/LT/LL-Breakdown ===
    group_w = bin_widths / 4
    ax2.bar(bin_centers - group_w, h_TL, width=group_w, color="mediumseagreen",
            label=f"TL (n={n_TL})")
    ax2.bar(bin_centers,           h_LT, width=group_w, color="steelblue",
            label=f"LT (n={n_LT})")
    ax2.bar(bin_centers + group_w, h_LL, width=group_w, color="indianred",
            label=f"LL (n={n_LL})")
    ax2.stairs(h_net, bins, color="black", lw=1.5, linestyle="--",
               label="net = TL+LT+LL (bkg_estimate_raw)")
    ax2.axhline(0.0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Events per Kategorie", fontsize=LABEL_SIZE)
    ax2.legend(fontsize=LEGEND_SIZE - 2, ncol=2)
    ax2.tick_params(labelsize=TICK_SIZE)

    # === Ax3: Bin-weise Non-closure ===
    colors = np.where(non_closure_per_bin >= 0, "forestgreen", "tomato")
    ax3.bar(
        bin_centers,
        non_closure_per_bin * 100,
        width=bin_widths,
        yerr=non_closure_per_bin_sigma * 100,
        capsize=3,
        ecolor="black",
        color=colors,
        alpha=0.7,
        label="bin-wise non-closure (± stat)"
    )
    ax3.axhline(0.0,  linestyle="--", color="gray", linewidth=0.8)
    ax3.axhline( abs(non_closure) * 100, linestyle=":", color="red",
                linewidth=0.8, label=f"±global ({non_closure*100:.1f} ± {sigma_non_closure*100:.1f}%)")
    ax3.axhline(-abs(non_closure) * 100, linestyle=":", color="red",
                linewidth=0.8)
    ax3.set_ylabel("Non-closure [%]", fontsize=LABEL_SIZE)
    ax3.set_xlabel("NN Score", fontsize=LABEL_SIZE)
    ax3.legend(fontsize=LEGEND_SIZE)
    ax3.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    plt.savefig(config.abcd_plot_path + f"matrix_cr_closure_{step_i:05d}.pdf")
    plt.close()
    return non_closure, non_closure_per_bin, sigma_non_closure

def unc_plot(config, opt_pars, data, scale, step_i):
    """Breaks down each region/sample's total uncertainty band into its
    individual systematic components (STAT, NORM, BKG_SHAPE, JET_PT, ...),
    so you can see which one actually dominates the band instead of just
    the combined nominal ± total."""
    hists = tomatos.pipeline.make_hists(
        opt_pars, data, config, scale, filter_return_hists=False, step=step_i,
    )
    for region in ["SR_btag_2", "VR_btag_2"]:
        if region not in hists:
            continue
        for sample in hists[region]:
            sample_hists = hists[region][sample]
            if "NOSYS" not in sample_hists:
                continue
            nom = np.array(sample_hists["NOSYS"])
            nom_sum = nom.sum()
            if nom_sum < 1e-10:
                continue

            # systematics that have both an _1UP and matching _1DOWN histogram
            sys_names = sorted({
                key[: -len("_1UP")]
                for key in sample_hists
                if key.endswith("_1UP") and key[: -len("_1UP")] + "_1DOWN" in sample_hists
            })
            if not sys_names:
                continue

            # per-bin relative size of each systematic (magnitude only - the
            # sign is fixed by convention: up stacks above the nominal
            # line, down stacks below it, regardless of whether an
            # individual bin's "up" variant happens to be below nominal)
            bins = np.array(config.bins)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_widths = np.diff(bins)

            colors = plt.cm.tab10(np.linspace(0, 1, len(sys_names)))

            fig, ax = plt.subplots(figsize=(8, 4.5))
            bottom_up = np.zeros_like(nom)
            bottom_down = np.zeros_like(nom)
            # quadrature sum, i.e. the statistically correct combination of
            # independent systematics - NOT the same as the stack height
            # (linear sum), which only shows relative contributions
            quad_up_sq = np.zeros_like(nom)
            quad_down_sq = np.zeros_like(nom)
            with np.errstate(invalid="ignore", divide="ignore"):
                for sys, color in zip(sys_names, colors):
                    up = np.array(sample_hists[f"{sys}_1UP"])
                    down = np.array(sample_hists[f"{sys}_1DOWN"])
                    rel_up = np.where(nom > 0, 100 * np.abs(up - nom) / nom, 0.0)
                    rel_down = np.where(nom > 0, 100 * np.abs(down - nom) / nom, 0.0)

                    ax.bar(bin_centers, rel_up, width=bin_widths, bottom=bottom_up,
                           color=color, edgecolor="white", linewidth=0.3, label=sys)
                    ax.bar(bin_centers, -rel_down, width=bin_widths, bottom=-bottom_down,
                           color=color, edgecolor="white", linewidth=0.3)
                    bottom_up += rel_up
                    bottom_down += rel_down
                    quad_up_sq += rel_up ** 2
                    quad_down_sq += rel_down ** 2

            ax.stairs(np.sqrt(quad_up_sq), bins, color="black", linewidth=1.5,
                      baseline=None, label="total (quadrature)")
            ax.stairs(-np.sqrt(quad_down_sq), bins, color="black", linewidth=1.5,
                      baseline=None)
            ax.axhline(0.0, color="black", linewidth=1.2)
            ax.set_xlabel("NN Score")
            ax.set_ylabel("Relative uncertainty [%]\n(stack = linear sum of parts, black = actual quadrature total)")
            ax.set_title(f"{region} / {sample} — uncertainty breakdown, step {step_i}")
            ax.legend(fontsize=9)
            fig.tight_layout()
            plt.savefig(
                config.unc_plot_path
                + f"unc_breakdown_{region}_{sample}_{step_i:05d}.pdf"
            )
            plt.close()



def plot_non_closure_history(config, steps, non_closures, non_closure_errs=None):
    """Plot non-closure (%) over all training steps collected every 10 steps."""
    steps = np.array(steps)
    vals = np.array(non_closures, dtype=float) * 100  # convert to %

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals, color="steelblue", linewidth=1.5)
    if non_closure_errs is not None:
        errs = np.array(non_closure_errs, dtype=float) * 100
        ax.fill_between(
            steps, vals - errs, vals + errs,
            alpha=0.25, color="steelblue", label="± stat",
        )
        ax.legend()
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Non-closure (%)")
    ax.set_title("Matrix-method CR non-closure over training")
    fig.tight_layout()
    plt.savefig(config.abcd_plot_path + "non_closure_history.pdf")
    plt.close()


def compute_asimov_significance(config, opt_pars, data, scale, step_i):
    hists = tomatos.pipeline.make_hists(
        opt_pars, data, config, scale, filter_return_hists=False, step=step_i,
    )
    sr = hists["SR_btag_2"]
    sig = np.array(sr[config.signal_sample]["NOSYS"])
    bkg = np.array(sr["bkg_estimate"]["NOSYS"])

    mask = bkg > 1e-10
    if not np.any(mask):
        return np.nan

    s = sig[mask]
    b = bkg[mask]
    # Cowan et al. bin-by-bin Asimov significance
    z_sq = 2.0 * np.sum((s + b) * np.log(1.0 + s / b) - s)
    return float(np.sqrt(max(z_sq, 0.0)))


def plot_asimov_history(config, steps, significances):
    steps = np.array(steps)
    vals = np.array(significances, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals, color="crimson", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$Z_A$")
    ax.set_title("Asimov significance over training")
    fig.tight_layout()
    plt.savefig(config.abcd_plot_path + "asimov_significance_history.pdf")
    plt.close()


def log_sr_comparison(config, opt_pars, data, scale, step_i):
    hists = tomatos.pipeline.make_hists(
        opt_pars, data, config, scale, filter_return_hists=False, step=step_i,
    )
    sr = hists["SR_btag_2"]

    sig = np.array(sr[config.signal_sample]["NOSYS"])
    bkg = np.array(sr["bkg"]["NOSYS"])
    est = np.array(sr["bkg_estimate"]["NOSYS"])

    sum_sig, sum_bkg, sum_est = sig.sum(), bkg.sum(), est.sum()
    if sum_sig < 1e-10 or sum_bkg < 1e-10 or sum_est < 1e-10:
        return

    sig_norm = sig / sum_sig
    bkg_norm = bkg / sum_bkg
    est_norm = est / sum_est

    # normiert
    sig_up_n, sig_down_n = _total_unc(sr[config.signal_sample], sig_norm, sum_sig)
    bkg_up_n, bkg_down_n = _total_unc(sr["bkg"],                bkg_norm, sum_bkg)
    est_up_n, est_down_n = _total_unc(sr["bkg_estimate"],       est_norm, sum_est)

    # unnormiert
    sig_up_a, sig_down_a = _total_unc(sr[config.signal_sample], sig, 1.0)
    bkg_up_a, bkg_down_a = _total_unc(sr["bkg"],                bkg, 1.0)
    est_up_a, est_down_a = _total_unc(sr["bkg_estimate"],       est, 1.0)

    entries = [
        (sig_norm, sig_up_n, sig_down_n, "crimson",     "-",  "signal"),
        (bkg_norm, bkg_up_n, bkg_down_n, "steelblue",   "-",  "bkg MC"),
        (est_norm, est_up_n, est_down_n, "forestgreen", "--", "bkg estimate"),
    ]
    entries_nom = [
        (sig, sig_up_a, sig_down_a, "crimson",     "-",  "signal"),
        (est, est_up_a, est_down_a, "forestgreen", "--", "bkg estimate"),
    ]

    bins = config.bins

    fig, ax = plt.subplots(figsize=(8, 5))

    for tag, en in [("normalized", entries), ("nominal", entries_nom)]:
        fig, ax = plt.subplots(figsize=(8, 5))

        for nom, s_up, s_down, color, ls, _ in en:
            ax.stairs(nom, bins, color=color, linestyle=ls, linewidth=1.5)
            ax.fill_between(
                bins, np.r_[s_down, s_down[-1]], np.r_[s_up, s_up[-1]],
                step="post", alpha=0.25, color=color,
            )

        handles = [
            (mpatches.Patch(color=c, alpha=0.25),
             mlines.Line2D([], [], color=c, linestyle=ls))
            for _, _, _, c, ls, _ in en
        ]
        labels = [f"{lbl} (± total unc)" for *_, lbl in en]
        ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})

        ax.set_xlabel("NN Score")
        ax.set_ylabel("Normalized Events" if tag == "normalized" else "Events")
        ax.set_title(f"SR_btag_2 Shape Comparison — step {step_i}")
        fig.tight_layout()
        fig.savefig(config.sr_plot_path + f"sr_comparison_{step_i:05d}_{tag}.pdf")
        plt.close(fig)



def log_bw(metrics, opt_pars):
    metrics["bw"] = opt_pars["bw"]
    logging.info(f"bw: {opt_pars['bw']}")


def log_cuts(config, opt_pars, metrics, infer_metrics_i):
    for var, cut_dict in config.opt_cuts.items():
        if cut_dict["keep"] == "window":
            center = opt_pars[f"cut_{var}_center"]
            half_width = np.exp(float(opt_pars[f"cut_{var}_logwidth"])) / 2
            lo = tomatos.utils.inverse_min_max_scale(config, center - half_width, cut_dict["idx"])
            hi = tomatos.utils.inverse_min_max_scale(config, center + half_width, cut_dict["idx"])
            logging.info(f"cut_{var}: window [{lo:.4g}, {hi:.4g}]")
            for suffix, val in [("_lo", lo), ("_hi", hi)]:
                key = f"cut_{var}{suffix}"
                if key not in metrics:
                    metrics[key] = []
                metrics[key] = val
                infer_metrics_i[key] = val
        else:
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
        if h_key.startswith("_"):
            continue
        metrics["h_" + h_key] = h
    for h_key, h in test_hists.items():
        if h_key.startswith("_"):
            continue
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