import datetime
import time

import cv2
import einops
import face_alignment
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decalib.datasets import detectors
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from omegaconf import DictConfig

import pasl.render
from pasl.core.model import PaslModel
from pasl.data import get_data_loader
from pasl.pae import PAE
from pasl.VGG19_LOSS import VGG19LOSS

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

vgg19 = VGG19LOSS().to(device)
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=True, device="cpu"
)


def angle_to_model_idx(a: float, b: float):
    a_is_s = a == 2 or a == 4 or a == 41 or a == 80 or a == 130 or a == 190
    a_is_f = a == 3 or a == 50 or a == 51 or a == 140
    a_is_p = (
        a == 1
        or a == 5
        or a == 10
        or a == 90
        or a == 110
        or a == 120
        or a == 200
        or a == 240
    )
    b_is_s = b == 2 or b == 4 or b == 41 or b == 80 or b == 130 or b == 190
    b_is_f = b == 3 or b == 50 or b == 51 or b == 140
    b_is_p = (
        b == 1
        or b == 5
        or b == 10
        or b == 90
        or b == 110
        or b == 120
        or b == 200
        or b == 240
    )

    # SS
    if a_is_s and b_is_s:
        return PAE.LABEL_TO_IDX["ss"]
    # FF
    if a_is_f and b_is_f:
        return PAE.LABEL_TO_IDX["ff"]
    # PP
    if a_is_p and b_is_p:
        return PAE.LABEL_TO_IDX["pp"]
    # FS/SF
    if (a_is_f and b_is_s) or (a_is_s and b_is_f):
        return PAE.LABEL_TO_IDX["fs"]
    # SP/PS
    if (a_is_s and b_is_p) or (a_is_p and b_is_s):
        return PAE.LABEL_TO_IDX["sp"]
    # FP/PF
    if (a_is_f and b_is_p) or (a_is_p and b_is_f):
        return PAE.LABEL_TO_IDX["fp"]


def train(cfg: DictConfig):
    # Load auxiliary models
    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)
    face_detector = detectors.FAN(device=device)
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

    # Setup dataloader
    train_loader = get_data_loader(
        root_path=cfg.val_list.root_path,
        list_path=cfg.val_list.path,
        img_size=cfg.model.img_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.globals.num_workers,
        drop_last=True,
        shuffle=True,
    )
    train_loader_iter = iter(train_loader)

    # Load models
    model = PaslModel(cfg)
    model_ema = PaslModel(cfg)
    model.to(device)
    model_ema.to(device)

    # Init model and copy to EMA version
    model.init()
    model_ema.load_state_dict(model.state_dict())

    optims = {
        key: torch.optim.Adam(
            params=getattr(model, key).parameters(),
            lr=cfg.train.lr,
            betas=[cfg.train.beta1, cfg.train.beta2],
            weight_decay=cfg.train.weight_decay,
        )
        for key in ["generator", "style_encoder", "discriminator", "discriminator2"]
    }

    def zero_grad():
        for key in optims:
            optims[key].zero_grad()

    if cfg.train.loss == "perceptual":
        # TODO: MAKE THIS WORK
        # criterion_id = LossEG(False, 0)
        criterion_id = None

    # # resume training if necessary
    # if args.resume_iter > 0:
    #     self._load_checkpoint(args.resume_iter)

    print("Start training...")
    start_time = time.time()

    for i in range(cfg.train.resume_iter, cfg.train.total_iters):
        try:
            minibatch = next(train_loader_iter)
        except Exception:
            train_loader_iter = iter(train_loader)
            minibatch = next(train_loader_iter)
            continue

        src, ref, gt, src_angle, ref_angle = minibatch
        src, ref, gt = src.to(device), ref.to(device), gt.to(device)
        src_angle, ref_angle = src_angle.to(device), ref_angle.to(device)

        # Convert to the format that DECA expects (RGB BCHW -> RGB BHWC * 255)
        src_cv2 = einops.rearrange(src, "b c h w -> b h w c")
        ref_cv2 = einops.rearrange(ref, "b c h w -> b h w c")
        src_cv2 = src_cv2 * 255
        ref_cv2 = ref_cv2 * 255

        # Render depth and landmarks
        embeds, tform, original_images = pasl.render.embeds_from_src_ref_imgs_batch(
            deca, face_detector, src_cv2, ref_cv2, device
        )
        depth, lm = pasl.render.depth_from_embeds(
            deca, embeds, tform, original_images, device
        )
        depth = einops.repeat(depth, "b 1 h w -> b 3 h w")
        depth = depth.to(device)
        lm = lm.to(device)

        # # DEBUG: output debug image of batch
        # images = [src, ref, lm, depth]
        # grid = einops.rearrange(images, "img b c h w -> c (b h) (img w)")
        # save_image(grid, "output/debug_grid.png")
        # print("Saved debug image.")

        if cfg.model.masks:
            masks = depth
        else:
            masks = None

        # train the discriminator
        d_loss, d_losses = compute_d_loss(
            model,
            cfg.train.lambda_reg,
            src,
            ref,
            depth,
            lm,
            masks=masks,
        )
        zero_grad()
        d_loss.backward()
        optims["discriminator"].step()
        optims["discriminator2"].step()

        # train the generator
        g_loss, g_losses = compute_g_loss(
            model,
            cfg,
            src,
            ref,
            src_angle,
            ref_angle,
            depth,
            lm,
            gt,
            criterion_id,
            masks=masks,
            pae=pae,
        )
        zero_grad()
        g_loss.backward()
        optims["generator"].step()
        optims["style_encoder"].step()

        # compute moving average of network parameters
        moving_average(model.generator, model_ema.generator, beta=0.999)
        moving_average(model.style_encoder, model_ema.style_encoder, beta=0.999)

        # print out log info
        if (i + 1) % cfg.train.print_every == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            log = "Elapsed time [%s], Iteration [%i/%i], " % (
                elapsed,
                i + 1,
                cfg.train.total_iters,
            )
            all_losses = dict()
            for loss, prefix in zip([d_losses, g_losses], ["D/", "G/"]):
                for key, value in loss.items():
                    all_losses[prefix + key] = value

            log += " ".join(
                ["%s: [%.4f]" % (key, value) for key, value in all_losses.items()]
            )
            print(log)

            with open("./expr/loss.txt", "a") as file:
                file.writelines(str(log) + "\n")
            for key, value in all_losses.items():
                # TODO: Replace with wandb
                # self.writer.add_scalar(key, value, i + 1)
                pass

        # # generate images for debugging
        # if (i + 1) % cfg.sample_every == 0:
        #     os.makedirs(cfg.sample_dir, exist_ok=True)
        #     utils.debug_image(nets_ema, cfg, inputs=inputs_val, step=i + 1)
        # # save model checkpoints
        # if (i + 1) % cfg.save_every == 0:
        #     self._save_checkpoint(step=i + 1)
        # # compute FID and LPIPS if necessary
        # if (i + 1) % cfg.eval_every == 0:
        #     calculate_metrics(nets_ema, cfg, i + 1, mode="latent")


def compute_d_loss(model, lambda_reg, src, ref, ref_depth, ref_lm, masks=None):
    ref.requires_grad_()
    ref_depth.requires_grad_()

    _, real_out_1 = model.discriminator(ref, ref_depth, ref_lm)
    _, real_out_2 = model.discriminator2(ref, ref_depth, ref_lm)

    real_out = real_out_1 + real_out_2
    loss_real = adv_loss(real_out, 1)
    loss_reg = r1_reg(real_out, ref)

    # with fake images
    with torch.no_grad():
        s_trg = model.style_encoder(src)
        x_fake = model.generator(ref_depth, ref_lm, s_trg, masks=masks)

    _, fake_out_1 = model.discriminator(x_fake, ref_depth, ref_lm)
    _, fake_out_2 = model.discriminator2(x_fake, ref_depth, ref_lm)

    fake_out = fake_out_1 + fake_out_2
    loss_fake = adv_loss(fake_out, 0)
    loss = loss_real + loss_fake + lambda_reg * loss_reg
    return loss, dict(real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item())


def compute_pae_loss():
    pass


def compute_g_loss(
    model,
    cfg,
    x1_source,
    x2_target,
    x1_angle,
    x2_angle,
    x2_target_lm,
    lm,
    gt,
    criterion_id,
    masks=None,
    facemodel=None,
    pae=None,
):
    # adversarial loss
    s_trg = model.style_encoder(x1_source)
    x_fake = model.generator(x2_target_lm, lm, s_trg, masks=masks)
    fake_fea_1, fake_out_1 = model.discriminator(x_fake, x2_target_lm, lm)
    fake_fea_2, fake_out_2 = model.discriminator2(x_fake, x2_target_lm, lm)

    out = fake_out_1 + fake_out_2
    loss_adv = adv_loss(out, 1)

    # VGG 19 recon loss
    if cfg.train.vgg19_recon_loss:
        vgg19_loss = torch.mean(vgg19(x_fake, gt))
    else:
        vgg19_loss = 0.0

    # l2 loss
    loss = nn.MSELoss()
    loss_l2 = loss(x_fake, gt)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pae.eval()
    loss_PAE = []
    x_fake_s = x_fake.detach().cpu().numpy()  # Convert to NumPy array (batch, C, H, W)
    x_fake_s = np.transpose(
        x_fake_s, (0, 2, 3, 1)
    )  # Change shape from (batch, C, H, W) to (batch, H, W, C)
    x_source_s = (
        x1_source.detach().cpu().numpy()
    )  # Convert to NumPy array (batch, C, H, W)
    x_source_s = np.transpose(
        x_source_s, (0, 2, 3, 1)
    )  # Change shape from (batch, C, H, W) to (batch, H, W, C)
    # Apply cv2 transformations to each image in the batch
    x_fake_s = np.array(
        [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_fake_s]
    )  # Convert RGB to BGR
    x_fake_s = np.array(
        [cv2.resize(img, (112, 112)) for img in x_fake_s]
    )  # Resize all images
    x_source_s = np.array(
        [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_source_s]
    )  # Convert RGB to BGR
    x_source_s = np.array(
        [cv2.resize(img, (112, 112)) for img in x_source_s]
    )  # Resize all images
    x_fake_s = (
        torch.from_numpy(x_fake_s).permute(0, 3, 1, 2).to(device)
    )  # Change shape back to (batch, C, H, W)
    x_source_s = (
        torch.from_numpy(x_source_s).permute(0, 3, 1, 2).to(device)
    )  # Change shape back to (batch, C, H, W)

    for i in range(x1_angle.shape[0]):
        face_encoder = pae.models[angle_to_model_idx(x1_angle[i], x2_angle[i])]
        real_embs = face_encoder(x_fake_s[i].unsqueeze(0))
        fake_embs = face_encoder(x_source_s[i].unsqueeze(0))
        loss_id = torch.mean(1 - cos(real_embs, fake_embs))
        loss_PAE.append(loss_id)

    loss_id_2 = torch.tensor(loss_PAE, device=device)
    loss_id_2 = loss_id_2.mean()

    # Style Consistency Loss
    if cfg.train.style_cyc:
        s_trg_2 = model.style_encoder(x_fake)
        loss_id_cyc = torch.mean(torch.abs(s_trg_2 - s_trg))
    else:
        loss_id_cyc = 0.0

    loss = (
        loss_adv
        + 10.0 * loss_id_2
        + 10.0 * loss_id_cyc
        + 10.0 * vgg19_loss
        + 50.0 * loss_l2
    )
    return loss, dict(
        total_loss=loss.item(),
        adv=loss_adv.item(),
        l2_loss=loss_l2.item(),
        id=loss_id_2.item(),
        id_cyc=loss_id_cyc.item(),
        vgg19_loss=vgg19_loss.item(),
    )


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


@hydra.main(version_base=None, config_path="../configs/", config_name="base_train")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
