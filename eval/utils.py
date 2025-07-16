import os

import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

from models.ir50 import IR_50


# Loads the eval images. RETURNS IN BGR
class EvalDataset(Dataset):
    def __init__(self, fake_dir, gt_dir, real_dir):
        self.fake_dir = fake_dir
        self.gt_dir = gt_dir
        self.real_dir = real_dir

        # Store the union of all the filenames
        # This is because corresponding generated fake image, gt image, and real image
        # must share the same filename.
        self.files = list(
            set(os.listdir(fake_dir))
            | set(os.listdir(gt_dir))
            | set(os.listdir(real_dir))
        )
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        fake_img_path = os.path.join(self.fake_dir, file)
        gt_img_path = os.path.join(self.gt_dir, file)
        real_img_path = os.path.join(self.real_dir, file)

        fake_img: torch.Tensor = read_image(fake_img_path)
        gt_img: torch.Tensor = read_image(gt_img_path)
        real_img: torch.Tensor = read_image(real_img_path)

        # (RGB, H, W) -> (BGR, H, W)
        # The models expect BGR (they were trained with images from cv.imread,
        # which outputs in BGR)
        fake_img = fake_img[[2, 1, 0], :, :]
        gt_img = gt_img[[2, 1, 0], :, :]
        real_img = real_img[[2, 1, 0], :, :]
        return file, fake_img, gt_img, real_img


def load_ir50(label: str, path: os.PathLike, device: torch.device):
    INPUT_SIZE = [112, 112]
    model = IR_50(INPUT_SIZE)

    print(f"Loading {label} model (MS1M-IR50) at '{path}'...")
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        raise Exception(f"Model not found at '{path}'")

    model.to(device)
    return model
