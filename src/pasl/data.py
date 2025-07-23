from pathlib import Path

import einops
import torch
import torch.linalg
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils import data
from torchvision import io, transforms


def read_rgb(path):
    img = io.read_image(str(path))
    # uint8->float, [0-255]->[0-1]
    img = img.to(torch.float32) / 255.0
    return img


# Crop image by 20px on the right, pad 10px on the top and resize into original.
# NOTE: Why is this done?
def preprocess(img: Float[Tensor, "c h w"]) -> Float[Tensor, "c h w"]:
    c, h, w = img.shape
    # To 256x256
    if (h, w) != (256, 256):
        img = F.interpolate(
            img.unsqueeze(0),  # need to add batch dim
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # Shave 20 pixels off the right edge image. Creates a 236x256 image
    cropped_img = img[:, :, :-20]
    # Pad top of image with 10 pixels of RGB
    img = F.pad(cropped_img, (0, 0, 10, 0), mode="constant", value=0)

    # Resize back to original size
    img = F.interpolate(
        img.unsqueeze(0),  # need to add batch dim
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return img


class SrcRefGtDepthLmAnglesDataset(data.Dataset):
    root_path: Path
    depth_lm_root_path: Path
    samples_src: list[Path]
    samples_ref: list[Path]
    samples_gt: list[Path]
    angles_src: list[int]
    angles_ref: list[int]

    def __init__(
        self,
        root_path,
        depth_lm_root_path,
        list_path,
        transform=None,
    ):
        self.root_path = Path(root_path)
        self.depth_lm_root_path = Path(depth_lm_root_path)
        self.samples_src = []
        self.samples_ref = []
        self.samples_gt = []
        self.angles_src = []
        self.angles_ref = []
        self.transform = transform

        # Check if depth-lm generated dir exists
        if not (self.depth_lm_root_path.exists() and self.depth_lm_root_path.is_dir()):
            raise Exception(
                f"Depth and landmark image directory not found at {self.depth_lm_root_path.absolute()}.\n"
                "Generate the images using the provided script `generate_depth_lm.py`"
            )

        print(f"Processing sample list at {list_path}")
        with open(list_path) as F:
            for line_num, line in enumerate(F):
                try:
                    line = line.strip("\n")
                    line_split = line.split(" ")
                    path1 = Path(line_split[0])
                    path2 = Path(line_split[1])
                    path3 = Path(line_split[2]) if len(line_split) > 2 else path2  # gt
                    basename1 = path1.name  # equiv to os.path.basename
                    basename2 = path2.name
                    # The angle of the subject in the image is included in the basename.
                    angle1 = int(basename1.split("_")[2])
                    angle2 = int(basename2.split("_")[2])
                except Exception:
                    print(f"Error reading dataset list on line {line_num}")
                    continue

                self.samples_src.append(path1)
                self.samples_ref.append(path2)
                self.samples_gt.append(path3)
                self.angles_src.append(angle1)
                self.angles_ref.append(angle2)
        print(f"Finished processing sample list at {list_path}")

    def __len__(self):
        return len(self.samples_src)

    def __getitem__(
        self, idx
    ) -> tuple[
        # Src sample (RGB)
        Float[Tensor, "3 256 256"],
        # Ref sample (RGB)
        Float[Tensor, "3 256 256"],
        # Ground truth sample (RGB)
        Float[Tensor, "3 256 256"],
        # Depth render
        Float[Tensor, "1 256 256"],
        # Landmarks (RGB)
        Float[Tensor, "3 256 256"],
        # source angle and ref angle (0-dim)
        Float[Tensor, ""],
        Float[Tensor, ""],
    ]:
        src = read_rgb(self.root_path / self.samples_src[idx])
        ref = read_rgb(self.root_path / self.samples_ref[idx])
        gt = read_rgb(self.root_path / self.samples_gt[idx])

        sample_idx = idx + 1
        pair_name = f"{sample_idx:04}.png"
        depth = read_rgb(self.depth_lm_root_path / "depth" / pair_name)
        lm = read_rgb(self.depth_lm_root_path / "lm" / pair_name)

        src_ang = torch.tensor(self.angles_src[idx], dtype=torch.long)
        ref_ang = torch.tensor(self.angles_ref[idx], dtype=torch.long)

        # NOTE: Why is this done?
        # depth = preprocess(depth)
        # lm = preprocess(lm)

        # Broadcast depth to RGB
        depth = einops.repeat(depth, "1 h w -> 3 h w")

        if self.transform is not None:
            src = self.transform(src)
            ref = self.transform(ref)
            gt = self.transform(gt)
            depth = self.transform(depth)
            lm = self.transform(lm)

        return (
            src,
            ref,
            gt,
            depth,
            lm,
            src_ang,
            ref_ang,
        )


def get_data_loader(
    root_path,
    depth_lm_root_path,
    list_path,
    img_size=256,
    batch_size=32,
    num_workers=0,
    drop_last=False,
):
    print("Preparing data loader...")

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ]
    )

    ds = SrcRefGtDepthLmAnglesDataset(
        root_path,
        depth_lm_root_path,
        list_path,
        transform=transform,
    )
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
