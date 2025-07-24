import einops
import torch
import torch.linalg
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from deca.decalib.datasets import datasets, detectors
from deca.decalib.deca import DECA


# Generate batches of depth and landmark images, based on source-reference pairs
# Returns (depth, batch)
def render_depth_lm_batch(
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
    assert len(src_path_list) == len(ref_path_list)
    src_td_og = datasets.TestData(src_path_list, face_detector, iscrop=True, sample_step=10)
    ref_td_og = datasets.TestData(ref_path_list, face_detector, iscrop=True, sample_step=10)
    
    # AoS -> SoA
    src_td = {"image": []}
    ref_td = {"image": [], "tform": [], "original_image": []}
    for td in src_td_og:
        src_td["image"].append(td["image"])
    for td in ref_td_og:
        ref_td["image"].append(td["image"])
        ref_td["tform"].append(td["tform"])
        ref_td["original_image"].append(td["original_image"])
    # Stack along batch dimension
    src_td["image"] = einops.pack(src_td["image"], "* c h w")[0].to(device)
    ref_td["image"] = einops.pack(ref_td["image"], "* c h w")[0].to(device)
    ref_td["tform"] = einops.pack(ref_td["tform"], "* rows cols")[0].to(device)
    ref_td["original_image"] = einops.pack(ref_td["original_image"], "* c h w")[0].to(
        device
    )

    with torch.no_grad():
        src_embeds = deca.encode(src_td["image"])
        ref_embeds = deca.encode(ref_td["image"])

        new_embeds = {}
        # the types of embeddings are
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

        # # DEBUG
        # x = torch.zeros(batchsize, device=device, dtype=torch.float32)
        # y = torch.arange(batchsize, device=device, dtype=torch.float32)
        # z = torch.zeros(batchsize, device=device, dtype=torch.float32)
        # cam_offset, _ = einops.pack([x, y, z], "batch *")
        # new_embeds["cam"] = ref_embeds["cam"] + 0.1 * cam_offset

        ref_tform_inv_t = torch.linalg.inv(ref_td["tform"]).transpose(-2, -1)
        opdict, visdict = deca.decode(
            new_embeds,
            render_orig=True,
            original_image=ref_td["original_image"],
            tform=ref_tform_inv_t,
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
    return depth, lm


def render_depth_lm_single(
    deca: DECA,
    face_detector: detectors.FAN,
    src_path: str,
    ref_path: str,
    device: torch.device | str,
) -> tuple[
    # Depth
    Float[Tensor, "1 256 256"],
    # Landmark
    Float[Tensor, "c 256 256"],
]:
    depth, lm = render_depth_lm_batch(
        deca, face_detector, [src_path], [ref_path], device
    )
    return depth.squeeze(0), lm.squeeze(0)
