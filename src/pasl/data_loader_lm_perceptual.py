"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from deca.decalib.datasets import datasets, detectors
from deca.decalib.deca import DECA
from deca.decalib.utils import util
from deca.decalib.utils.config import cfg as deca_cfg


class LMDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        list_path,
        transform=None,
        train_data="mpie",
        multi=False,
        device="cpu",
    ):
        self.device = device
        self.deca = DECA(config=deca_cfg, device=self.device)
        self.face_detector = detectors.FAN(device=self.device)
        self.multi = multi
        self.train_data = train_data
        self.transform = transform
        self.root_dir = root_dir

        if multi:
            self.samples = []
            self.samples2 = []
            self.samples3 = []
            self.samples4 = []
            self.samples5 = []
            self.samples6 = []
            self.samples7 = []
            self.samples8 = []
            self.samples9 = []

            with open(list_path) as F:
                for line in F:
                    line = line.strip("\n")
                    # print(line.split(' '))
                    self.samples.append(line.split(" ")[0])
                    self.samples2.append(line.split(" ")[1])
                    self.samples3.append(line.split(" ")[2])
                    self.samples4.append(line.split(" ")[3])
                    self.samples5.append(line.split(" ")[4])
                    self.samples6.append(line.split(" ")[5])
                    self.samples7.append(line.split(" ")[6])
                    self.samples8.append(line.split(" ")[7])
                    self.samples9.append(line.split(" ")[8])

        else:
            self.samples = []
            self.samples_angle = []
            self.samples2 = []
            self.samples2_angle = []
            self.samples3 = []

            with open(list_path) as F:
                n = 0
                try:
                    for line in F:
                        line = line.strip("\n")

                        # print(line.split(" "))
                        self.samples.append(line.split(" ")[0])
                        self.samples_angle.append(
                            int(line.split(" ")[0].split("/")[-1].split("_")[2])
                        )
                        self.samples2.append(line.split(" ")[1])
                        self.samples2_angle.append(
                            int(line.split(" ")[1].split("/")[-1].split("_")[2])
                        )
                        try:
                            gt_path = line.split(" ")[2]
                            self.samples3.append(gt_path)
                        except:
                            self.samples3.append(line.split(" ")[1])

                except Exception:
                    n += 1
                    print("error {} images".format(n))
                    del self.samples[-1]
                    del self.samples2[-1]
                    pass

            # print(self.samples[0:5])
            # print(self.samples_angle[0:5])
            # print(self.samples2[0:5])
            # print(self.samples2_angle[0:5])
            # print(self.samples3[0:5])

    def __getitem__(self, index):
        if self.multi:
            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]
            fname3_folder = self.samples3[index]
            fname4_folder = self.samples4[index]
            fname5_folder = self.samples5[index]
            fname6_folder = self.samples6[index]
            fname7_folder = self.samples7[index]
            fname8_folder = self.samples8[index]
            fname9_folder = self.samples9[index]

            if self.train_data == "vox1":
                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                fname4 = fname4_folder
                fname5 = fname5_folder
                fname6 = fname6_folder
                fname7 = fname7_folder
                fname8 = fname8_folder
                fname9 = fname9_folder

                fname_lm9 = (
                    fname9.split("unzippedFaces")[0]
                    + "lm/unzippedFaces"
                    + fname9.split("unzippedFaces")[1]
                )

            img = Image.open(fname).convert("RGB")
            img2 = Image.open(fname2).convert("RGB")
            img3 = Image.open(fname3).convert("RGB")
            img4 = Image.open(fname4).convert("RGB")
            img5 = Image.open(fname5).convert("RGB")

            img6 = Image.open(fname6).convert("RGB")
            img7 = Image.open(fname7).convert("RGB")
            img8 = Image.open(fname8).convert("RGB")
            img9 = Image.open(fname9).convert("RGB")

            img_lm9 = Image.open(fname_lm9).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
                img4 = self.transform(img4)
                img5 = self.transform(img5)
                img6 = self.transform(img6)
                img7 = self.transform(img7)
                img8 = self.transform(img8)
                img9 = self.transform(img9)

                img_lm9 = self.transform(img_lm9)
                # img_lm = self.transform(img_lm)

            return img, img2, img3, img4, img5, img6, img7, img8, img9, img_lm9

        else:
            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]
            fname3_folder = self.samples3[index]
            fname_angle = self.samples_angle[index]
            fname2_angle = self.samples2_angle[index]

            if self.train_data == "mpie":
                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                name_angle = fname_angle
                name2_angle = fname2_angle
            elif self.train_data == "vox1":
                fname = fname_folder
                fname2 = fname2_folder
                fname = fname.split("\t")[0].replace("\\", "/")
                fname2 = fname2.replace("\\", "/")

                lm = (
                    fname2.split("vox1_full_face_crop_256")[0]
                    + "LM_256"
                    + fname2.split("vox1_full_face_crop_256")[1]
                )
            elif self.train_data == "vox2":
                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                name_angle = fname_angle
                name2_angle = fname2_angle

                lm = (
                    fname2.split("crop_256")[0] + "LM_256" + fname2.split("crop_256")[1]
                )

            fname = os.path.join(self.root_dir, fname)
            fname2 = os.path.join(self.root_dir, fname2)
            fname3 = os.path.join(self.root_dir, fname3)
            img = Image.open(fname).convert("RGB")
            img2 = Image.open(fname2).convert("RGB")
            gt = Image.open(fname3).convert("RGB")
            img_lm2, lm = get_depth_render(
                self.deca, self.face_detector, fname, fname2, self.device
            )
            img_lm2 = img_lm2.resize((256, 256))
            img_lm = img_lm2

            flattened_width = img_lm2.width - 20
            flattened_width1 = lm.width - 20
            # flattened_height= img_lm2.height+20
            # 壓扁照片
            new_image = Image.new(img_lm2.mode, (flattened_width, img_lm2.height))
            new_image1 = Image.new(lm.mode, (flattened_width1, lm.height))
            # 複製原始圖像到新圖像的中間，加上 margin
            new_image.paste(img_lm2, (0, 10))
            new_image1.paste(lm, (0, 10))
            # 將新圖像重新調整為目標尺寸
            # flattened_image = img_lm2 .resize(( flattened_width, flattened_height))
            # plt.imshow(flattened_image)
            # flattened_image1 = lm .resize(( flattened_width, img_lm2.height))
            # 將壓扁後的照片重新調整為目標尺寸
            img_lm2 = new_image.resize((224, 224), Image.ANTIALIAS)

            lm = new_image1.resize((256, 256), Image.ANTIALIAS)

            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
                img_lm2 = self.transform(img_lm2)
                img_lm = self.transform(img_lm)
                lm = self.transform(lm)
                gt = self.transform(gt)

            print(img.shape)
            print(img2.shape)
            print(img_lm2.shape)
            print(img_lm.shape)
            print(gt.shape)

            return (
                img,
                img2,
                img_lm,
                img_lm2,
                lm,
                gt,
                torch.tensor(name_angle),
                torch.tensor(name2_angle),
            )

    def __len__(self):
        return len(self.samples)


def get_depth_render(deca, face_detector, src_path, ref_path, device):
    testdata = datasets.TestData(
        [src_path, ref_path],
        face_detector,
        iscrop=True,
        sample_step=10,
    )
    i = 0
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    # deca_cfg.rasterizer_type = "pytorch3d"

    src = testdata[i]["image"].to(device)[None, ...]
    ref = testdata[i + 1]["image"].to(device)[None, ...]

    with torch.no_grad():
        codedict1 = deca.encode(src)
        codedict2 = deca.encode(ref)
        src_shape = codedict1["shape"]

        light_code = codedict1["light"]
        tex_code = codedict1["tex"]
        detail_code = codedict1["detail"]

        ref_shape = codedict2["shape"]
        temp = codedict2
        temp["shape"] = src_shape
        temp["light"] = light_code
        temp["tex"] = tex_code
        temp["detail"] = detail_code
        tform = testdata[i + 1]["tform"][None, ...]
        tform = torch.inverse(tform).transpose(1, 2).to(device)
        original_image = testdata[i + 1]["original_image"][None, ...].to(device)
        orig_opdict, orig_visdict = deca.decode(
            temp, render_orig=True, original_image=original_image, tform=tform
        )
        orig_visdict["inputs"] = original_image
        # cv2.imwrite('1.png', cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256)))
        lm_image = cv2.resize(
            util.tensor2image(orig_visdict["landmarks2d"][0]), (256, 256)
        )
        depth_image = deca.render.render_depth(orig_opdict["trans_verts"]).repeat(
            1, 3, 1, 1
        )[0]

        depth_image = depth_image.detach().cpu().numpy()
        depth_image = depth_image * 255.0
        depth_image = np.maximum(np.minimum(depth_image, 255), 0)
        depth_image = depth_image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
        depth_image = Image.fromarray(np.uint8(depth_image))
        lm_image = Image.fromarray(lm_image)

    return depth_image, lm_image


def get_eval_loader_vgg(
    root_dir,
    list_path,
    img_size=256,
    batch_size=32,
    imagenet_normalize=True,
    num_workers=0,
    drop_last=False,
    train_data="mpie",
    multi=False,
    device="cpu",
):
    print("Preparing DataLoader for the evaluation phase...")
    if imagenet_normalize:
        height, width = 299, 299

    else:
        height, width = img_size, img_size

    transform = transforms.Compose(
        [
            transforms.Resize([img_size, img_size]),
            transforms.Resize([height, width]),
            transforms.ToTensor(),
        ]
    )

    dataset = LMDataset(
        root_dir,
        list_path,
        transform=transform,
        train_data=train_data,
        multi=multi,
        device=device,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
