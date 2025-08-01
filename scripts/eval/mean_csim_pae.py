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

device = torch.device("cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


@torch.no_grad()
def calculate_csim_for_all_tasks(dataloader):
    print("Loading PAE model...")
    # Order must be the same as in LABEL_TO_INDEX

    pae = PAE()
    pae.load_from_paths(
        dict(
            ff="./weights/pae/ff_backbone_ir_50_epoch_80.pth",
            fs="./weights/pae/fs_backbone_ir_50_epoch_80.pth",
            fp="./weights/pae/fp_backbone_ir_50_epoch_120.pth",
            ss="./weights/pae/ss_backbone_ir_50_epoch_100.pth",
            sp="./weights/pae/sp_backbone_ir_50_epoch_150.pth",
            pp="./weights/pae/pp_backbone_ir_50_epoch_100.pth",
        )
    )
    pae.to(device)
    pae.eval()

    fake_gt_csim_list = []
    fake_real_csim_list = []
    real_gt_csim_list = []
    ard_list = []

    resize = T.Resize((112, 112), T.InterpolationMode.BILINEAR, antialias=True)

    error_count_total = 0

    # 遍历两个文件夹中的图像并计算旋转误差
    for fake_img, gt_img, real_img in tqdm(dataloader, desc="Processing"):
        batchsize = fake_img.shape[0]
        fake_img = fake_img.to(device, torch.float32)
        gt_img = gt_img.to(device, torch.float32)
        real_img = real_img.to(device, torch.float32)

        fake_img = resize(fake_img)
        gt_img = resize(gt_img)
        real_img = resize(real_img)

        fake_angle, fake_valid = pae.img_to_angle(fake_img)
        gt_angle, gt_valid = pae.img_to_angle(gt_img)
        real_angle, real_valid = pae.img_to_angle(real_img)
        valid = fake_valid & gt_valid & real_valid
        error_count_total += batchsize - valid.size(0)

        import plotly.express as px

        # # DEBUG:
        # invalid_debug_img = einops.rearrange(
        #     [
        #         real_img[~valid],
        #         gt_img[~valid],
        #         fake_img[~valid],
        #     ],
        #     "column row c h w -> (row h) (column w) c",
        # )
        # invalid_debug_img = invalid_debug_img * 255
        # invalid_debug_img = invalid_debug_img.to("cpu", torch.uint8).numpy()
        # fig = px.imshow(invalid_debug_img)
        # fig.show()

        # Calculate ARD
        ard = torch.abs(fake_angle[valid] - gt_angle[valid])
        ard_list.append(ard)

        # Embeddings
        angles_2, _ = einops.pack([fake_angle, real_angle], "b *")
        model_idx_to_batches = pae.angles_to_processing_list(angles_2, valid)
        fake_embs = pae(fake_img, model_idx_to_batches)
        real_embs = pae(real_img, model_idx_to_batches)
        gt_embs = pae(gt_img, model_idx_to_batches)

        # only use valid ones
        fake_embs = fake_embs[valid]
        real_embs = real_embs[valid]
        gt_embs = gt_embs[valid]

        # Cosine similarity
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
    print("ard:", torch.cat(ard_list).mean().item())
    print("error count:", error_count_total)


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
