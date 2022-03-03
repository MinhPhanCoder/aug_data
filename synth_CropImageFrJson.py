from synth_utils.utils_load_data import CustomDataset
import numpy as np
from synth_utils import utils_synth
from synth_highlight import extract_bboxes
import cv2 as cv
from PIL import Image


if __name__ == "__main__":
    CustomDS = CustomDataset()
    CustomDS.load_Custom(r"datatset/input/backgrounds")
    CustomDS.prepare()
    for image_id in CustomDS.image_ids:
        image = CustomDS.load_image(image_id)
        mask, class_ids = CustomDS.load_mask(image_id)
        bbox = extract_bboxes(mask)
        for i in range(mask.shape[-1]):
            m = mask.astype('uint8')[:, :, i]*255
            # m = cv.cvtColor(m, cv.COLOR_GRAY2BGR)
            b = bbox[i]
            a = cv.bitwise_and(image[b[0]:b[2], b[1]:b[3]], image[b[0]:b[2], b[1]:b[3]], mask = m[b[0]:b[2], b[1]:b[3]])
            a[np.where((a==[0,0,0]).all(axis=2))] =[255,255,255]
            utils_synth.show_wait_destroy("img", a)
            a = Image.fromarray(np.uint8(a)).convert('RGBA')
            a.save("100.png")
            exit(-1)