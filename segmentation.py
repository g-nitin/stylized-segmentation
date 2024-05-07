from pathlib import Path

import torch
import numpy as np
from typing import List
from streamlit import cache_data
from utils import get_model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def init():
    """Initialize the models."""

    model_dir = 'models'
    model_name = "sam_vit_b_01ec64.pth"
    model_file = Path(model_dir, model_name)  # Use Path object

    response = get_model(model_name, 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                         model_dir)
    if not response[0]:  # Check if the file downloaded successfully
        print(response[1])
        exit()

    # Downloading SAM has been moved to `requirements.txt`

    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_type: str = "vit_b"

    return model_file, model_type, device


@cache_data(ttl=10*60, show_spinner="Performing segmentation")
def perform_segmentation(uploaded_image, num_masks, model_file, model_type, device) -> List[np.ndarray]:
    sam = sam_model_registry[model_type](checkpoint=model_file).to(device=torch.device(device))

    # Automatic mask generation
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate a list of dictionaries describing individual segmentations
    output_mask = mask_generator.generate(uploaded_image)

    # Sort the segments by their area
    sorted_masks = sorted(output_mask, key=(lambda x: x['area']), reverse=True)

    out_masks = []  # List to hold the top `num_masks` by area
    for i, val in enumerate(sorted_masks):
        if i + 1 > num_masks:
            break
        # Append the mask (a value corresponding to 'segmentation' key)
        out_masks.append(val['segmentation'])

    # The return type contains are boolean masks of the original image's shape
    return out_masks
