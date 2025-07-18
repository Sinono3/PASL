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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets, self.nets_ema = build_model(cfg)
        self.writer = SummaryWriter("output/log/test_reconstruction")

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + "_ema", module)

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ("ema" not in name) and ("fan" not in name):
                print("Initializing %s..." % name)
                network.apply(utils.he_init)

    # Loads nets_ema from a path
    def load_from_path(self, path):
        pickle = torch.load(path, map_location=self.device)
        self.nets_ema.generator = pickle['generator']
        self.nets_ema.style_encoder = pickle['style_encoder']
