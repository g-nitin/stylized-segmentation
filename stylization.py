from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from adain import adaptive_instance_normalization, coral
import adain_net


# Adapted from https://github.com/naoto0804/pytorch-AdaIN/tree/master

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, device, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def stylize(content: Image, style: Image,
            vgg_path: str = 'models/vgg_normalised.pth', decoder_path: str = 'models/decoder.pth',
            content_size: int = 0, style_size: int = 0, crop: bool = False,
            output_dir: Union[str, Path, None] = None, output_file_name: str = 'stylized_mask.png',
            preserve_color: bool = True, alpha: float = 1.0) -> torch.Tensor:
    """
    :param content: The Image object representing the content image.
    :param style: The Image object representing the style image.
    :param vgg_path: The path to the vgg model. Default 'models/vgg_normalised.pth'.
    :param decoder_path: The path to the decoder model. 'models/decoder.pth'
    :param content_size: The (minimum) size for the content image, keeping the original size if set to 0 (default).
    :param style_size: The (minimum) size for the style image, keeping the original size if set to 0 (default).
    :param crop: Boolean to center crop to create square image. Default False.
    :param output_dir: If specified, then the directory to save the output images. Default None
    :param output_file_name: The filename for the output image, with the specified extensions.
        Default 'stylized_mask.jpg'. Note that the stylized image will be saved at `output_dir`->`output_file_name`.
    :param preserve_color: Boolean to preserve color of the content image. Default True
    :param alpha: The weight that controls the degree of stylization. Should be between 0 and 1 (default).
    :return: torch.Tensor representing the final image.
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = adain_net.decoder
    decoder.eval()
    decoder.load_state_dict(torch.load(decoder_path))
    decoder.to(device)

    vgg = adain_net.vgg
    vgg.eval()
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    content = content_tf(content)
    style = style_tf(style)

    if preserve_color:
        style = coral(style, content)

    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, device, alpha, None)
    output = output.cpu()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_name = output_dir.joinpath(output_file_name)
        save_image(output, str(output_name))
        # print(f'Stylized image saved at {output_name}')

    return output
