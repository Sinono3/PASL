from pathlib import Path

import torch
import torch.linalg
from jaxtyping import Float, UInt8
from torch import Tensor
from torch.utils import data
from torchvision import io, transforms


def read_rgb(path) -> Float[Tensor, "c h w"]:
    img = io.read_image(str(path), io.ImageReadMode.RGB)
    # uint8->float, [0-255]->[0-1]
    img = img.to(torch.float32) / 255.0
    return img


class SrcRefGtAnglesDataset(data.Dataset):
    root_path: Path
    samples_src: list[Path]
    samples_ref: list[Path]
    samples_gt: list[Path]
    angles_src: list[int]
    angles_ref: list[int]

    def __init__(
        self,
        root_path,
        list_path,
        transform=None,
    ):
        self.root_path = Path(root_path)
        self.samples_src = []
        self.samples_ref = []
        self.samples_gt = []
        self.angles_src = []
        self.angles_ref = []
        self.transform = transform

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
        # source angle and ref angle (0-dim)
        UInt8[Tensor, ""],
        UInt8[Tensor, ""],
        # Src sample path
        str,
        # Ref sample path
        str,
        # Ground truth sample path
        str,
    ]:
        src_path = self.root_path / self.samples_src[idx]
        ref_path = self.root_path / self.samples_ref[idx]
        gt_path = self.root_path / self.samples_gt[idx]

        src = read_rgb(src_path)
        ref = read_rgb(ref_path)
        gt = read_rgb(gt_path)

        src_ang = torch.tensor(self.angles_src[idx], dtype=torch.uint8)
        ref_ang = torch.tensor(self.angles_ref[idx], dtype=torch.uint8)

        if self.transform is not None:
            src = self.transform(src)
            ref = self.transform(ref)
            gt = self.transform(gt)

        return (
            src,
            ref,
            gt,
            src_ang,
            ref_ang,
            str(src_path),
            str(ref_path),
            str(gt_path),
        )


def get_data_loader(
    root_path,
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

    ds = SrcRefGtAnglesDataset(
        root_path,
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
