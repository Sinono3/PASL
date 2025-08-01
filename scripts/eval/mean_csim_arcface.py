from pathlib import Path

import einops
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from pasl.eval.utils import EvalDataset
from pasl.pae import PAE
from pasl.utils import set_seed
from pasl.ir50 import load_ir50

device = torch.device("cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


@torch.no_grad()
def calculate_csim_for_all_tasks(dataloader):
    criterion = load_ir50(
        "Arcface", "./weights/arcface/backbone_ir50_ms1m_epoch63.pth", device
    )
    criterion.to(device)
    criterion.eval()

    fake_gt_csim_list = []
    fake_real_csim_list = []
    real_gt_csim_list = []

    resize = T.Resize((112, 112), T.InterpolationMode.BILINEAR, antialias=True)

    # 遍历两个文件夹中的图像并计算旋转误差
    for fake_img, gt_img, real_img in tqdm(dataloader, desc="Processing"):
        fake_img = fake_img.to(device, torch.float32)
        gt_img = gt_img.to(device, torch.float32)
        real_img = real_img.to(device, torch.float32)

        # Resize and normalize image to [0-1]
        fake_img = resize(fake_img)
        gt_img = resize(gt_img)
        real_img = resize(real_img)

        with torch.torch.no_grad():
            fake_embs = criterion(fake_img)
            gt_embs = criterion(gt_img)
            real_embs = criterion(real_img)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        fake_gt_csim = cos(fake_embs, gt_embs)
        fake_real_csim = cos(fake_embs, real_embs)
        real_gt_csim = cos(real_embs, gt_embs)

        fake_gt_csim_list.append(fake_gt_csim)
        fake_real_csim_list.append(fake_real_csim)
        real_gt_csim_list.append(real_gt_csim)

    print("fake_gt_csim:", torch.cat(fake_gt_csim_list).mean().item())
    print("fake_real_csim:", torch.cat(fake_real_csim_list).mean().item())
    print("real_gt_csim:", torch.cat(real_gt_csim_list).mean().item())


@hydra.main(version_base=None, config_path="../../configs/", config_name="base_eval")
def main(cfg: DictConfig):
    set_seed(cfg.globals.seed)
    dataset = EvalDataset(cfg.list.root_path, cfg.list.path, cfg.eval.fake_dir)
    dataloader = DataLoader(
        dataset, batch_size=cfg.eval.batch_size, num_workers=cfg.globals.num_workers
    )
    calculate_csim_for_all_tasks(dataloader)


if __name__ == "__main__":
    main()
