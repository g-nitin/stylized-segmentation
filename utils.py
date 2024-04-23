import os
from PIL import Image
from typing import Union
import numpy as np
from pathlib import Path
from shutil import rmtree


def combine_with_mask(content_path: Union[str, Path], style_path: Union[str, Path],
                      masked_array: np.ndarray, save_path: Union[str, Path, None] = None) -> Image.Image:
    """
    :param content_path: The path to the original content image
    :param style_path: The path to the styled segmented image
    :param masked_array: The mask array corresponding to the segment
    :param save_path:
    :return:
    """
    # Load the two image files
    image1 = Image.open(content_path)
    image2 = Image.open(style_path)

    # Load the boolean numpy array
    boolean_array = (~masked_array).copy()

    # Convert boolean array to integer (0s and 1s)
    boolean_array = boolean_array.astype(np.uint8)

    # Convert boolean array to a mask
    mask = Image.fromarray(boolean_array * 255)

    # Resize mask to match image dimensions
    mask = mask.resize(image1.size, Image.Resampling.LANCZOS)

    # Convert mask to alpha channel
    image1.putalpha(mask)

    # Invert mask to use with image2
    inverted_mask = Image.fromarray((1 - boolean_array) * 255)
    inverted_mask = inverted_mask.resize(image2.size, Image.Resampling.LANCZOS)
    image2.putalpha(inverted_mask)

    # Ensure both images have the same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)

    # Combine images
    combined_image = Image.alpha_composite(image1, image2)

    # Save or display the combined image
    if save_path:
        combined_image.save(save_path)
        print(f'Final image saved at {save_path}')

    return combined_image


def create_folder(folder_name):
    """
    Creates a new folder with the specified name.

    :param folder_name: The desired name for the folder.
    :return: The path to the created folder.
    """
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def delete_folder(folder_path):
    """
    Deletes the specified folder and its contents.

    :param folder_path: The path to the folder to be deleted.
    """
    rmtree(folder_path)  # Recursive
