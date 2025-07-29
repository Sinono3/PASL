import argparse
import threading
import time
from typing import Iterable, Union

import cv2
import einops
import face_alignment
import gradio as gr
import hydra
import numpy as np
import torch
import torchvision.io
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from skimage import transform as trans
from torch import Tensor
from torchvision import transforms

import pasl.render
from deca.decalib.datasets import datasets, detectors
from deca.decalib.deca import DECA
from deca.decalib.utils import util
from deca.decalib.utils.config import cfg as deca_cfg
from pasl.solver_lm_perceptual import Solver

deca = None
solver = None
fa = None
face_detector = None


def np_hwc255_to_tensor(img: np.ndarray) -> Float[Tensor, "c h w"]:
    assert isinstance(img, np.ndarray), (
        "the img type is {}, but ndarray expected".format(type(img))
    )

    # Try to convert BGR to RGB. If it fails, the tensor is (most likely) already in RGB
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img = img

    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return (img.to(torch.float32) / 255.0).unsqueeze(0)


def tensor_to_np_hwc255(
    tensor: Float[Tensor, "b c h w"],
) -> Float[np.ndarray, "b h w c"]:
    # [0,1] -> [0,255]
    img = (tensor * 255.0).to("cpu", torch.uint8)
    #  CHW -> HWC, torch->numpy
    return img.permute(0, 2, 3, 1).numpy()


def align_face(rimg, landmarks):
    # only care about the x,y components
    landmarks = [[p[0], p[1]] for p in landmarks[0]]

    src = np.array(
        [
            [30.2946 * 2, 51.6963 * 2],
            [65.5318 * 2, 51.5014 * 2],
            [48.0252 * 2, 71.7366 * 2],
            [33.5493 * 2, 92.3655 * 2],
            [62.7299 * 2, 92.2041 * 2],
        ],
        dtype=np.float32,
    )
    src[:, 0] += 31.9721

    landmarks = np.array(landmarks)
    assert landmarks.shape[0] == 68 or landmarks.shape[0] == 5
    assert landmarks.shape[1] == 2
    if landmarks.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmarks[36] + landmarks[39]) / 2
        landmark5[1] = (landmarks[42] + landmarks[45]) / 2
        landmark5[2] = landmarks[30]
        landmark5[3] = landmarks[48]
        landmark5[4] = landmarks[54]
    else:
        landmark5 = landmarks

    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img_crop = cv2.warpAffine(rimg, M, (256, 256), borderValue=0.0)
    return img_crop


# -----------------------------------------------------------------------------------
# multi thread
class ipcamCapture:
    def __init__(self, url):
        self.Frame = []
        self.status = False
        self.isstop = False

        # 攝影機連接。
        self.capture = cv2.VideoCapture(url)

    def start(self):
        # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print("ipcam started!")
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        # 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print("ipcam stopped!")

    def getframe(self):
        # 當有需要影像時，再回傳最新的影像。
        return self.Frame.copy()

    def queryframe(self):
        while not self.isstop:
            self.status, self.Frame = self.capture.read()

        self.capture.release()


topil = transforms.ToPILImage()
frompil = transforms.ToTensor()


# -----------------------------------------------------------------------------------
def main_face_video(video):
    print("Processing video...")
    device = "cuda"

    # Must be of size 4, to make a 2x2 grid
    src_image_paths = [
        "./examples/96.jpg",
        "./examples/287.jpg",
        "./examples/1466_01_03_0018.jpg",
        "./examples/1465_01_03_0055.jpg",
    ]

    img_size = 256
    transform = transforms.Resize([img_size, img_size])

    # input img
    src = [
        torchvision.io.read_image(src_path, torchvision.io.ImageReadMode.RGB)
        for src_path in src_image_paths
    ]
    src, _ = einops.pack(
        src,
        "* channel height width",
    )

    src = transform(src).to(device, torch.float32)
    src = src / 255.0

    # DEBUG: output src images before processing
    # grid = [src[i] for i in range(4)]
    # grid = einops.rearrange(grid, "(h1 w1) c h w -> c (h1 h) (w1 w)", h1=2, w1=2)
    # yield topil(grid), topil(grid)

    src_embed = solver.extract(src)

    idx = 0
    freq = 2

    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        idx += 1
        ret = cap.grab()
        if idx % freq == 0:
            ret, ref_frame_bgr = cap.retrieve()

            landmarks = fa.get_landmarks(ref_frame_bgr)
            cropped_ref = align_face(ref_frame_bgr, landmarks)
            cropped_ref = torch.tensor(cropped_ref, device=device, dtype=torch.float32)

            embeds, tform, original_image = pasl.render.embeds_from_imgs(
                deca, face_detector, [cropped_ref], device
            )
            depth, lm = pasl.render.depth_from_embeds(
                deca, embeds, tform, original_image, device
            )

            # Depth -> repeat 3xDepth, 4xBatch
            depth = einops.repeat(depth, "1 1 h w -> 4 3 h w")
            # LM -> 4xBatch
            lm = einops.repeat(lm, "1 c h w -> 4 c h w")

            output = solver.sample(src_embed, depth, lm)

            # DEBUG: don't feed depth/LM into model
            # zero_depth = torch.zeros_like(depth)
            # zero_lm = torch.zeros_like(lm)
            # output = solver.sample(src_embed, zero_depth, zero_lm)

            # HxWxC BGR [0-255] -> CxHxW RGB [0-1]
            cropped_ref = einops.rearrange(cropped_ref, "h w c -> c h w")
            cropped_ref = cropped_ref[[2, 1, 0], :, :] / 255.0

            # Convert to grid
            output = einops.rearrange(
                output, "(h1 w1) c h w -> c (h1 h) (w1 w)", h1=2, w1=2
            )

            # DEBUG: output depth/LM
            # depth = depth[0]
            # lm = lm[0]
            # yield topil(depth), topil(lm)

            # To PIL
            yield topil(cropped_ref), topil(output)


# # ---------------------------------------------------------------------------------------------------------


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = colors.orange,
        secondary_hue: Union[colors.Color, str] = colors.fuchsia,
        neutral_hue: Union[colors.Color, str] = colors.blue,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="url('https://fastly.picsum.photos/id/213/4928/3264.jpg?hmac=OC0yPL-iiM1YgVjpAjbMf51MjnR6cycgmn1TSLJhDZ0') no-repeat center/cover",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            # button_secondary_text_color="blue",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="6px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )


def serve_gradio():
    css = """
    #image_upload{min-height:400px}
    #image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
    """

    seafoam = Seafoam()

    with gr.Blocks(theme=seafoam, css=css) as demo:
        with gr.Row():
            gr.Markdown("""# Webcam Input""")
            gr.Markdown("# Source Image")
            gr.Markdown("# Transformed Faces")
        with gr.Row():
            crop_frames = gr.Image(height=480, width=480)
            source_image = gr.Image("./examples/musk.png", height=480, width=480)
            output_video = gr.Image(label="output", height=480, width=480)
        with gr.Row():
            with gr.Column():
                gr.Markdown("""# MP4 Input""")
                input_video = gr.Video(label="video input", value="examples/Tom.mp4")
            with gr.Column():
                gr.Markdown("""# Buttons""")
                with gr.Row():
                    run_webcam_btn = gr.Button("Run (webcam)")
                    run_btn = gr.Button("Run")
                    stop_btn = gr.Button(value="Stop", variant="primary")
                with gr.Column():
                    gr.Markdown("""# Examples""")
                    with gr.Row():
                        examples = gr.Examples(
                            [
                                "examples/Tom.mp4",
                                "examples/wick.mp4",
                                "examples/jennacut.mp4",
                            ],
                            inputs=input_video,
                            label=" ",
                        )

        # cam_event = run_wecam_btn.click(
        #     main_face_cam, input_video, [crop_frames, output_video]
        # )
        # stop.click(fn=None, inputs=None, outputs=None, cancels=[ex_event, cam_event])
        # output_video.edit(None, None, None, cancels=[cam_event, ex_event])
        ex_event = run_btn.click(
            main_face_video,
            input_video,
            [crop_frames, output_video],
        )
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[ex_event])
        output_video.edit(None, None, None, cancels=[ex_event])

    demo.queue()
    demo.launch(debug=True)


@hydra.main(version_base=None, config_path="../../configs/", config_name="base_demo")
def main(cfg: DictConfig):
    global solver, fa, deca, face_detector
    solver = Solver(cfg, "cuda")
    solver.load_from_path(cfg.model.nets_ema_path)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=True, device="cuda"
    )
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device="cuda")
    face_detector = detectors.FAN()

    # DEBUG:
    # crop_frames, output_video = next(main_face_video("examples/Tom.mp4"))
    # print(crop_frames, output_video)

    serve_gradio()


if __name__ == "__main__":
    main()
