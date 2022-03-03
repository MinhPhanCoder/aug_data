from PIL import Image, ImageEnhance
import numpy as np
import random
import imageio
from imgaug import augmenters as iaa


def _transform_foreground(fg_path):
    """"
        Hàm aug đơn giản
    """
    fg_image = Image.open(fg_path)
    fg_alpha = np.array(fg_image.getchannel(3))
    assert np.any(fg_alpha == 0), f'ảnh foreground phải có kênh màu alpha: {str(fg_path)}'
    # Rotate the foreground
    angle_degrees = random.randint(0, 359)
    fg_image = fg_image.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)
    # Scale the foreground
    scale = random.random() * .1 + .5  # Pick something between .5 and 1
    new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
    fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)
    # Adjust foreground brightness
    brightness_factor = random.random() * .4 + .7  # Pick something between .7 and 1.1
    enhancer = ImageEnhance.Brightness(fg_image)
    fg_image = enhancer.enhance(brightness_factor)
    # Add any other transformations here...
    return fg_image


def aug_img_fg_custom(fg_path):
    """"
        Hàm aug cho foreground
    """
    img = imageio.imread(fg_path)
    seq_pipeline = iaa.Sequential([
        iaa.Resize((0.3, 1)),  # Resize
        iaa.Flipud(0.5),  # Flip
        iaa.Affine(rotate=(-180, 180)),  # Nghiêng
        iaa.PiecewiseAffine(scale=(0.01, 0.3)),  # Biến dạng ảnh, giống distort
        iaa.ScaleX((0.5, 1.5)),  # Thu phóng theo trục X
        iaa.ScaleY((0.5, 1.5)),  # Thu phóng theo trục y
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 3.0))), # Blur
        iaa.AddElementwise((-10, 10)),
        iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}),
    ])
    img_aug = seq_pipeline(image=img)
    return Image.fromarray(np.uint8(img_aug)).convert('RGBA')


def aug_img_bg_custom(bg_path):
    """"
        Hàm aug cho background
    """
    img = imageio.imread(bg_path)
    seq_pipeline = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.WithBrightnessChannels(iaa.Add((-10, 10)))),
        iaa.Sometimes(0.5, iaa.AddToBrightness((-20, 20))),
        iaa.Sometimes(0.3, iaa.MultiplyHueAndSaturation((0.5, 2.5), per_channel=True)),
        iaa.Sometimes(0.3, iaa.MultiplyHue((0.5, 2.5))),
        iaa.Sometimes(0.3, iaa.MultiplySaturation((0.5, 2.5))),
    ])
    img_aug = seq_pipeline(image=img)
    return Image.fromarray(np.uint8(img_aug)).convert('RGBA')
