from functools import partial

import matplotlib.pyplot as plt
import optax
from jaxopt import OptaxSolver

import tomatos.pipeline
import tomatos.utils


def build_lr_schedule(config):
    """Build the base LR schedule (without plateau reduction)."""
    if "linear_cycle" in config.lr_schedule:
        return optax.linear_onecycle_schedule(
            transition_steps=config.num_steps,
            peak_value=config.lr,
            pct_start=0.3,
            div_factor=50,
            final_div_factor=1,
            pct_final=0.9,
        )
    elif "constant" in config.lr_schedule:
        if hasattr(config, "bce_warmup_steps") and "cls" in config.objective:
            return optax.join_schedules(
                [
                    optax.constant_schedule(config.lr),
                    optax.constant_schedule(config.lr * config.cls_lr_factor),
                ],
                boundaries=[config.bce_warmup_steps],
            )
        else:
            return optax.constant_schedule(config.lr)
    raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")


def setup(config, pars, lr_schedule=None):
    if lr_schedule is None:
        lr_schedule = build_lr_schedule(config)

    learning_rates = [lr_schedule(i) for i in range(config.num_steps)]

    # nice to plot this right away
    plt.figure(figsize=(6, 5))
    plt.plot(learning_rates)
    plt.yscale("log")
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(config.plot_path + "lr_schedule.pdf")
    plt.close()

    # successively apply gradient updates for gradient transformations with
    # optax.chain
    # https://optax.readthedocs.io/en/latest/api/combining_optimizers.html

    # mask gradient updates only for passed vars
    def mask(pars: dict, vars: list):
        return {key: key in vars for key in pars}

    # limiting bandwidth and cut updates is important to avoid gradient
    # explosion for these
    if config.objective == "bce":
        optimizer = optax.chain(
            optax.adam(lr_schedule),
            optax.masked(
                optax.set_to_zero(),  # no update for cuts in bce
                mask(pars, ["bins"]),
            ),
            optax.masked(
                optax.set_to_zero(),  # no update for cuts in bce
                mask(pars, [key for key in pars.keys() if "cut_" in key]),
            ),
            optax.add_decayed_weights(1e-4)
        )
    else:
        optimizer = optax.chain(
            optax.zero_nans(),  # if nans, zero out, otherwise opt breaks entirely
            optax.clip_by_global_norm(1.0),  # prevent gradient explosion → NaN cascade
            optax.adam(lr_schedule),
            optax.masked(
                optax.add_decayed_weights(1e-3),  # only NN weights, not cuts/bw
                mask(pars, ["nn"]),
            ),
            # optax.add_noise(eta=0.001, gamma=0.5, seed=0),
            optax.masked(
                optax.clip(max_delta=config.update_limit_bw),
                mask(pars, ["bw"]),
            ),
            optax.masked(
                optax.set_to_zero() if (not config.include_cuts and config.cuts_start_step is None) else optax.clip(max_delta=config.update_limit_cuts),
                mask(pars, [key for key in pars.keys() if "cut_" in key and not key.endswith("_logwidth")]),
            ),
            # cut_*_logwidth lives in log-space (half_width = exp(logwidth)/2)
            # so it needs its own step-size limit, separate from the linear
            # [0,1]-scaled cut_*_center/cut_* limit above
            optax.masked(
                optax.set_to_zero() if (not config.include_cuts and config.cuts_start_step is None) else optax.clip(max_delta=config.update_limit_cut_width),
                mask(pars, [key for key in pars.keys() if key.endswith("_logwidth")]),
            ),

        )

    # has_aux allows, to return additional values from loss_fn than just the
    # loss value
    # dont jit the tomatos.pipeline.loss_fn, only literally jits this function
    # and will fail in the current setup
    return OptaxSolver(tomatos.pipeline.loss_fn, opt=optimizer, has_aux=True, jit=False)