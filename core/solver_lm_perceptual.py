"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from os.path import join as ospj

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import core.utils_lm as utils
from core.checkpoint import CheckpointIO
from core.model_lm_talking import build_model


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets, self.nets_ema = build_model(args)
        self.writer = SummaryWriter("log/test_reconstruction")

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + "_ema", module)

        if args.mode == "eval":
            self.ckptios = [
                CheckpointIO(
                    ospj(
                        args.checkpoint_dir, "{}_nets_ema.ckpt".format(args.resume_iter)
                    ),
                    **self.nets_ema,
                )
            ]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ("ema" not in name) and ("fan" not in name):
                print("Initializing %s..." % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)
