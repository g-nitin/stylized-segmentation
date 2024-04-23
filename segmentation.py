from pathlib import Path

import torch
import numpy as np
from os.path import exists, join
from subprocess import run
from typing import List
from streamlit import cache_data


def init():
    Path('./models').mkdir(exist_ok=True)

    # Define the URL and the local file path
    # !wget - nc -P. / models / https: // dl.fbaipublicfiles.com / segment_anything / sam_vit_h_4b8939.pth
    # !wget - nc - P. / models / https: // dl.fbaipublicfiles.com / segment_anything / sam_vit_b_01ec64.pth
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    model_file = join(".", "models", "sam_vit_b_01ec64.pth")

    # Check if the file already exists
    if not exists(model_file):
        run(["wget", "-nc", "-P", "./models/", url])

    # Also download SAM
    # pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    run(['pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'])

    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_type: str = "vit_b"

    return model_file, model_type, device


@cache_data(ttl=10*60, show_spinner="Performing segmentation")
def perform_segmentation(uploaded_image, num_masks, model_file, model_type, device) -> List[np.ndarray]:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[model_type](checkpoint=model_file).to(device=torch.device(device))

    # Automatic mask generation
    mask_generator = SamAutomaticMaskGenerator(sam)

    output_mask = mask_generator.generate(uploaded_image)
    sorted_masks = sorted(output_mask, key=(lambda x: x['area']), reverse=True)

    out_masks = []
    for i, val in enumerate(sorted_masks):
        if i + 1 > num_masks:
            break
        out_masks.append(val['segmentation'])

    return out_masks
