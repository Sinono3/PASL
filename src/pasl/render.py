import einops
import torch
import torch.linalg
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from deca.decalib.datasets import datasets, detectors
from deca.decalib.deca import DECA


# Combines embeds from source and reference image paths.
# Returns: new embeds, and 3x3 matrices for transformation to use when rendering.
def embeds_from_src_ref_paths(
    deca: DECA,
    face_detector: detectors.FAN,
    src_path_list: list[str],
    ref_path_list: list[str],
    device: torch.device | str,
) -> tuple[
    # Embeddings
    Float[Tensor, "batch embed"],
    # Transforms (3x3)
    Float[Tensor, "batch rows cols"],
    # Original images (RGB)
    Float[Tensor, "batch channel height width"],
]:
    assert len(src_path_list) == len(ref_path_list)
    src_td_og = datasets.TestData(src_path_list, face_detector, iscrop=True)
    ref_td_og = datasets.TestData(ref_path_list, face_detector, iscrop=True)
    return embeds_from_src_ref_td(deca, face_detector, src_td_og, ref_td_og, device)


# Combines embeds from source and reference images.
# Returns: new embeds, and 3x3 matrices for transformation to use when rendering.
def embeds_from_src_ref_imgs(
    deca: DECA,
    face_detector: detectors.FAN,
    src_imgs: list[Float[Tensor, "height width channel"]],
    ref_imgs: list[Float[Tensor, "height width channel"]],
    device: torch.device | str,
) -> tuple[
    # Embeddings
    Float[Tensor, "batch embed"],
    # Transforms (3x3)
    Float[Tensor, "batch rows cols"],
    # Original images (RGB)
    Float[Tensor, "batch channel height width"],
]:
    assert len(src_imgs) == len(ref_imgs)
    src_td = [
        datasets.TestData.img_to_td(src_img, face_detector, iscrop=True)
        for src_img in src_imgs
    ]
    ref_td = [
        datasets.TestData.img_to_td(ref_img, face_detector, iscrop=True)
        for ref_img in ref_imgs
    ]
    return embeds_from_src_ref_td(deca, face_detector, src_td, ref_td, device)


# Combines embeds from a single image (no merging of source, reference)
# Returns: new embeds, and 3x3 matrices for transformation to use when rendering.
def embeds_from_imgs(
    deca: DECA,
    face_detector: detectors.FAN,
    # Should be in BGR
    imgs: list[Float[Tensor, "height width channel"]],
    device: torch.device | str,
) -> tuple[
    # Embeddings
    Float[Tensor, "batch embed"],
    # Transforms (3x3)
    Float[Tensor, "batch rows cols"],
    # Original images (RGB)
    Float[Tensor, "batch channel height width"],
]:
    td_og = [
        datasets.TestData.img_to_td(
            img.cpu().numpy(),
            face_detector,
            iscrop=True,
        )
        for img in imgs
    ]

    # AoS -> SoA
    td = {}
    # Stack tensors along batch dimension
    td["image"] = einops.pack([td["image"] for td in td_og], "* c h w")[0].to(device)  # fmt: off
    td["tform"] = einops.pack([td["tform"] for td in td_og], "* rows cols")[0].to(device)  # fmt: off
    td["original_image"] = einops.pack([td["original_image"] for td in td_og], "* c h w")[0].to(device)  # fmt: off

    with torch.no_grad():
        embeds = deca.encode(td["image"])

    tform_inv_t = torch.linalg.inv(td["tform"]).transpose(-2, -1)
    return embeds, tform_inv_t, td["original_image"]


# Combines embeds from source and reference image in TestData format.
# Returns: new embeds, and 3x3 matrices for transformation to use when rendering.
def embeds_from_src_ref_td(
    deca: DECA,
    face_detector: detectors.FAN,
    src_td_og: datasets.TestData | list[dict],
    ref_td_og: datasets.TestData | list[dict],
    device: torch.device | str,
) -> tuple[
    # Embeddings
    Float[Tensor, "batch embed"],
    # Transforms (3x3)
    Float[Tensor, "batch rows cols"],
    # Original images (RGB)
    Float[Tensor, "batch channel height width"],
]:
    assert len(src_td_og) == len(ref_td_og)

    # AoS -> SoA
    src_td = {}
    ref_td = {}
    # Stack tensors along batch dimension
    src_td["image"] = einops.pack([td["image"] for td in src_td_og], "* c h w")[0].to(device)  # fmt: off
    ref_td["image"] = einops.pack([td["image"] for td in ref_td_og], "* c h w")[0].to(device)  # fmt: off
    ref_td["tform"] = einops.pack([td["tform"] for td in ref_td_og], "* rows cols")[0].to(device)  # fmt: off
    ref_td["original_image"] = einops.pack([td["original_image"] for td in ref_td_og], "* c h w")[0].to(device)  # fmt: off

    with torch.no_grad():
        src_embeds = deca.encode(src_td["image"])
        ref_embeds = deca.encode(ref_td["image"])

    new_embeds = {}
    # the types of embeddings are:
    # ['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail']
    # We take `shape`, `tex`, `light`, `detail` from the source
    new_embeds["shape"] = src_embeds["shape"]
    new_embeds["tex"] = src_embeds["tex"]
    new_embeds["light"] = src_embeds["light"]
    new_embeds["detail"] = src_embeds["detail"]
    # Use the other embeddings from the references.
    new_embeds["exp"] = ref_embeds["exp"]
    new_embeds["pose"] = ref_embeds["pose"]
    new_embeds["cam"] = ref_embeds["cam"]
    new_embeds["images"] = ref_embeds["images"]

    # # DEBUG: add offset to camera according to batch index
    # x = torch.zeros(batchsize, device=device, dtype=torch.float32)
    # y = torch.arange(batchsize, device=device, dtype=torch.float32)
    # z = torch.zeros(batchsize, device=device, dtype=torch.float32)
    # cam_offset, _ = einops.pack([x, y, z], "batch *")
    # new_embeds["cam"] = ref_embeds["cam"] + 0.1 * cam_offset

    # Prepare transformation for rendering
    ref_tform_inv_t = torch.linalg.inv(ref_td["tform"]).transpose(-2, -1)
    return new_embeds, ref_tform_inv_t, ref_td["original_image"]


def depth_from_embeds(
    deca: DECA,
    embeds: Float[Tensor, "batch embed"],
    tform: Float[Tensor, "batch rows cols"],
    original_images: Float[Tensor, "batch channel height width"],
    device: torch.device | str,
):
    opdict, visdict = deca.decode(
        embeds,
        render_orig=True,
        original_image=original_images,
        tform=tform,
    )

    depth: Float[Tensor, "b c h w"] = deca.render.render_depth(
        opdict["trans_verts"]
    ).to(device)
    lm: Float[Tensor, "b c h w"] = visdict["landmarks2d"].to(device)
    # Clamp into [0,1]
    depth = depth.clamp(0, 1)
    lm = lm.clamp(0, 1)
    # Resize depth to 256x256
    depth = F.interpolate(depth, size=(256, 256), mode="bilinear", align_corners=False)

    # # DEBUG: output color instead of landmarks
    # return depth, visdict["shape_detail_images"].to(device)
    return depth, lm


# Generate batches of depth and landmark images, based on source-reference pairs
# Returns (depth, batch)
def depth_from_src_ref_paths(
    deca: DECA,
    face_detector: detectors.FAN,
    src_path_list: list[str],
    ref_path_list: list[str],
    device: torch.device | str,
) -> tuple[
    # Depth
    Float[Tensor, "b 1 256 256"],
    # Landmark
    Float[Tensor, "b c 256 256"],
]:
    embeds, render_tform, original_images = embeds_from_src_ref_paths(
        deca, face_detector, src_path_list, ref_path_list, device
    )
    return depth_from_embeds(deca, embeds, render_tform, original_images, device)
