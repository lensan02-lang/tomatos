#!/usr/bin/env python3
import argparse
import json
import logging

import equinox as eqx
import jax
import numpy as np

import tomatos.config
import tomatos.plotting
import tomatos.preprocess
import tomatos.training
import tomatos.utils

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
parser.add_argument("--prep", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--plot", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
# handy for scanning parameters from cli
parser.add_argument("--aux", type=float, default=1)

args = parser.parse_args()


def main():
    config = tomatos.config.Setup(args)

    if config.plot_inputs:
        logging.info("Plotting Inputs...")
        tomatos.plotting.plot_inputs(config)

    if args.prep:
        logging.info("Preprocessing...")
        tomatos.preprocess.run(config)
    if args.train:
        logging.info("Training...")
        tomatos.training.run(config)
    if args.plot:
        logging.info(f"Plotting here: {config.plot_path}")
        tomatos.plotting.model_plots(config)