import os
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

from pasl.data import read_rgb


# Loads the eval images. RETURNS IN BGR
class EvalDataset(Dataset):
    samples_real: list[Path]
    samples_gt: list[Path]
    samples_fake: list[Path]

    def __init__(self, root_dir, list_path, fake_dir):
        self.root_path = Path(root_dir)
        self.fake_dir = Path(fake_dir)

        self.samples_real = []
        self.samples_gt = []
        self.samples_fake = []

        print(f"Processing sample list at {list_path}")
        with open(list_path) as F:
            for line_num, line in enumerate(F):
                try:
                    line = line.strip("\n")
                    line_split = line.split(" ")
                    src_path = Path(line_split[0])
                    ref_path = Path(line_split[1])
                except Exception:
                    print(f"Error reading dataset list on line {line_num}")
                    continue

                sample_idx = line_num + 1
                # NOTE: only support for one output, currently.
                output_idx = 1
                output_basename = f"{sample_idx:04}_{output_idx:02}.png"

                abs_src_path = self.root_path / src_path
                abs_fake_path = self.fake_dir / output_basename
                abs_ref_path = self.root_path / ref_path
                # print(f"{abs_src_path}: {abs_src_path.exists()}")
                # print(f"{abs_fake_path}: {abs_fake_path.exists()}")
                # print(f"{abs_ref_path}: {abs_ref_path.exists()}")

                if (
                    abs_src_path.is_file()
                    and abs_fake_path.is_file()
                    and abs_ref_path.is_file()
                ):
                    self.samples_real.append(src_path)
                    self.samples_fake.append(output_basename)
                    self.samples_gt.append(ref_path)

    def __len__(self):
        return len(self.samples_real)

    def __getitem__(self, idx):
        real_img: torch.Tensor = read_rgb(self.root_path / self.samples_real[idx])
        fake_img: torch.Tensor = read_rgb(self.fake_dir / self.samples_fake[idx])
        gt_img: torch.Tensor = read_rgb(self.root_path / self.samples_gt[idx])

        # (RGB, H, W) -> (BGR, H, W)
        # The models expect BGR (they were trained with images from cv.imread,
        # which outputs in BGR)
        fake_img = fake_img[[2, 1, 0], :, :]
        gt_img = gt_img[[2, 1, 0], :, :]
        real_img = real_img[[2, 1, 0], :, :]
        return fake_img, gt_img, real_img
