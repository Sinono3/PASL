"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
import pathlib

import torch
from tqdm import tqdm

from core import utils_lm
from core.data_loader_lm_perceptual import get_eval_loader_vgg


@torch.no_grad()
def generate_images(
    nets,
    cfg,
    mode: "eval",
    list_path: os.PathLike,
    output_dir: os.PathLike,
    output_label: str,
):
    output_dir = pathlib.Path(cfg.output_dir) / "eval" / output_label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read the testing image
    loader_eval = get_eval_loader_vgg(
        root=cfg.dataset.root_path,
        list_path=cfg.dataset.eval_list_path,
        img_size=cfg.model.img_size,
        batch_size=cfg.batch_size,
        imagenet_normalize=False,
        drop_last=True,
        mode="eval",
    )

    if os.path.exists(os.path.join(output_dir)):
        print("Output directory already exists. Aborting.")
        return

    path_fake = output_dir / "fake"
    path_real = output_dir / "real"
    path_real_lm = output_dir / "real_lm"
    path_ground_truth_lm = output_dir / "ground_truth_lm"

    shutil.rmtree(path_fake, ignore_errors=True)
    shutil.rmtree(path_real, ignore_errors=True)
    shutil.rmtree(path_real_lm, ignore_errors=True)
    shutil.rmtree(path_ground_truth_lm, ignore_errors=True)

    path_fake.mkdir(parents=True, exist_ok=True)
    path_real.mkdir(parents=True, exist_ok=True)
    path_real_lm.mkdir(parents=True, exist_ok=True)
    path_ground_truth_lm.mkdir(parents=True, exist_ok=True)

    print("Generating images ...")
    for i, x_src in enumerate(tqdm(loader_eval, total=len(loader_eval))):
        lm = x_src[4]
        x2_target_lm = x_src[3]
        x2_target = x_src[1]

        lm = lm.to(device)
        x2_target_lm = x2_target_lm.to(device)
        x2_target = x2_target.to(device)

        N = x2_target_lm.size(0)  # batch-size
        if cfg.model.masks:
            masks = x2_target_lm
        else:
            masks = None

        for j in range(cfg.num_outputs_per_domain):
            x1_source = x_src[0]
            x1_source = x1_source.to(device)
            s_trg = nets.style_encoder(x1_source)
            x_fake = nets.generator(x2_target_lm, lm, s_trg, masks=masks)

            # save generated images to calculate FID later
            for k in range(N):
                idx1 = i * cfg.batch_size + (k + 1)
                idx2 = j + 1
                basename = "%.4i_%.2i.png" % (idx1, idx2)
                filename = path_fake / basename
                filename2 = path_real / basename
                filename3 = path_real_lm / basename
                filename4 = path_ground_truth_lm / basename

                utils_lm.save_image(x_fake[k], ncol=1, filename=filename)
                utils_lm.save_image(x1_source[k], ncol=1, filename=filename2)
                utils_lm.save_image(x2_target_lm[k], ncol=1, filename=filename3)
                utils_lm.save_image(x2_target[k], ncol=1, filename=filename4)
