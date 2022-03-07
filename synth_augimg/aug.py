import imageio
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa


def aug_foreground(path_foreground):
    img = imageio.imread(path_foreground)
    seq_pipeline = iaa.Sequential([
        iaa.Resize((0.3, 1)),
        iaa.Flipud(0.5),
        """
        iaa.Affine(rotate=(-180, 180)),
        iaa.PiecewiseAffine(scale=(0.01, 0.3)),
        iaa.ScaleX((0.5, 1.5)),
        iaa.ScaleY((0.5, 1.5)),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 3.0))),
        iaa.AddElementwise((-10, 10)),
        iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}),
        """
    ])
    img_aug = seq_pipeline(image=img)
    return Image.fromarray(np.uint8(img_aug)).convert('RGBA')


def aug_background(path_background):
    img = imageio.imread(path_background)
    seq_pipeline = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.WithBrightnessChannels(iaa.Add((-10, 10)))),
        iaa.Sometimes(0.5, iaa.AddToBrightness((-20, 20))),
        """
        iaa.Sometimes(0.3, iaa.MultiplyHueAndSaturation((0.5, 2.5), per_channel=True)),
        iaa.Sometimes(0.3, iaa.MultiplyHue((0.5, 2.5))),
        iaa.Sometimes(0.3, iaa.MultiplySaturation((0.5, 2.5))),
        """
    ])
    img_aug = seq_pipeline(image=img)
    return Image.fromarray(np.uint8(img_aug)).convert('RGBA')
