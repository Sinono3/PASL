import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.utils import EvalDataset, load_ir50

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


def calculate_csim_for_all_tasks(fake_dir, gt_dir, real_dir):
    criterion = load_ir50(
        "Arcface", "./weights/arcface/backbone_ir50_ms1m_epoch63.pth", device
    )
    dataset = EvalDataset(fake_dir, gt_dir, real_dir)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

    fake_gt_csim_list = []
    fake_real_csim_list = []
    real_gt_csim_list = []
    # 遍历两个文件夹中的图像并计算旋转误差
    for fake_img, gt_img, real_img in tqdm(dataloader, desc="Processing"):
        fake_img = fake_img.to(device, torch.float32)
        gt_img = gt_img.to(device, torch.float32)
        real_img = real_img.to(device, torch.float32)

        criterion.eval()
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


if __name__ == "__main__":
    fake_image_dir = "./output/eval/mpie/250000"
    gt_image_dir = "./output/eval/mpie/250000ground_truth"
    real_image_dir = "./output/eval/mpie/250000real"

    calculate_csim_for_all_tasks(fake_image_dir, gt_image_dir, real_image_dir)
