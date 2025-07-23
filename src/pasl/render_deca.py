import einops
import torch
import torch.linalg
from torch import Tensor
from jaxtyping import Float

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
    Float[Tensor, "b 1 224 224"],
    # Landmark
    Float[Tensor, "b c 256 256"],
]:
    assert len(src_path_list) == len(ref_path_list)
    batchsize = len(src_path_list)

    td = datasets.TestData(
        src_path_list + ref_path_list, face_detector, iscrop=True, sample_step=10
    )

    # Recover data from datasets.TestData
    src_td = {"image": []}
    ref_td = {"image": [], "tform": [], "original_image": []}
    for i in range(0, batchsize):
        src_td["image"].append(td[i]["image"])
    for i in range(batchsize, batchsize * 2):
        ref_td["image"].append(td[i]["image"])
        ref_td["tform"].append(td[i]["tform"])
        ref_td["original_image"].append(td[i]["original_image"])

    # Stack along batch dimension
    src_td["image"] = einops.pack(src_td["image"], "* c h w")[0].to(device)
    ref_td["image"] = einops.pack(ref_td["image"], "* c h w")[0].to(device)
    ref_td["tform"] = einops.pack(ref_td["tform"], "* rows cols")[0].to(device)
    ref_td["original_image"] = einops.pack(ref_td["original_image"], "* c h w")[0].to(
        device
    )

    ref_tform_inv_t = torch.linalg.inv(ref_td["tform"]).transpose(-2, -1)
    ref_original = ref_td["original_image"]

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

        opdict, visdict = deca.decode(
            new_embeds,
            render_orig=True,
            original_image=ref_original,
            tform=ref_tform_inv_t,
        )

    depth: Float[torch.Tensor, "b c h w"] = deca.render.render_depth(
        opdict["trans_verts"]
    ).to(device)
    lm: Float[torch.Tensor, "b c h w"] = visdict["landmarks2d"].to(device)

    # Clamp into [0,1]
    depth = depth.clamp(0, 1)
    lm = lm.clamp(0, 1)
    return depth, lm


def render_depth_lm_single(
    deca: DECA,
    face_detector: detectors.FAN,
    src_path: str,
    ref_path: str,
    device: torch.device | str,
) -> tuple[
    # Depth
    Float[Tensor, "1 224 224"],
    # Landmark
    Float[Tensor, "c 256 256"],
]:
    td = datasets.TestData(
        [src_path, ref_path], face_detector, iscrop=True, sample_step=10
    )
    src = td[0]["image"].to(device)
    ref = td[1]["image"].to(device)

    # Matrix inverse is slow.
    # TODO: Since the transform we get here is a similarity
    # (https://nvision-user-guide.readthedocs.io/en/latest/geometric_transformations.html#similarity)
    # we could compute the inverse in a cheaper way.
    # But since it's 3x3, it won't matter that much.
    tform_inv_t = torch.linalg.inv(td[1]["tform"].to(device)).transpose(-2, -1)
    original = td[1]["original_image"].to(device)

    with torch.no_grad():
        src_embed = deca.encode(src.unsqueeze(0))
        ref_embed = deca.encode(ref.unsqueeze(0))

        new_embed = {}
        # the types of embeddings are
        # ['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail']
        # We take `shape`, `tex`, `light`, `detail` from the source
        new_embed["shape"] = src_embed["shape"]
        new_embed["tex"] = src_embed["tex"]
        new_embed["light"] = src_embed["light"]
        new_embed["detail"] = src_embed["detail"]
        # Use the other embeddings from the reference.
        new_embed["exp"] = ref_embed["exp"]
        new_embed["pose"] = ref_embed["pose"]
        new_embed["cam"] = ref_embed["cam"]
        new_embed["images"] = ref_embed["images"]

        opdict, visdict = deca.decode(
            new_embed,
            render_orig=True,
            original_image=original.unsqueeze(0),
            tform=tform_inv_t.unsqueeze(0),
        )

    depth: Float[torch.Tensor, "b c h w"] = deca.render.render_depth(
        opdict["trans_verts"]
    ).to(device)
    lm: Float[torch.Tensor, "b c h w"] = visdict["landmarks2d"].to(device)

    # Remove batch dimension
    depth = depth.squeeze(0)
    lm = lm.squeeze(0)

    # Clamp into [0,1]
    depth = depth.clamp(0, 1)
    lm = lm.clamp(0, 1)
    return depth, lm
