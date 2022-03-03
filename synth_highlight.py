from synth_utils.utils_load_data import CustomDataset
import numpy as np
from synth_utils import utils_visualize
import argparse
import cv2 as cv
from pathlib import Path


def extract_bboxes(mask):
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def get_parse_hl():
    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True,
                        help="Thư mục chứa ảnh cần highlight")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True,
                        help="Thư mục chứa ảnh sau khi highlight")
    parser.add_argument("--count", type=int, dest="count", default=10,
                        help="Số lượng cần highlight")
    return parser


def highlight_img(args):
    CustomDS = CustomDataset()
    CustomDS.load_Custom(Path(args.input_dir))
    CustomDS.prepare()
    for image_id in CustomDS.image_ids[0:args.count]:
        image = CustomDS.load_image(image_id)
        mask, class_ids = CustomDS.load_mask(image_id)
        bbox = extract_bboxes(mask)
        print("image_id ", image_id, CustomDS.image_reference(image_id))
        # img_instance = utils_visualize.display_instances(image, bbox, mask, class_ids,
        #                                                 CustomDS.class_names)
        fn = str(Path(args.output_dir) / CustomDS.image_info[image_id]["id"])
        utils_visualize.display_instances(image, bbox, mask, class_ids,
                                                         CustomDS.class_names, path_save=fn)
        #cv.imwrite(fn, img_instance)


if __name__ == "__main__":
    args = get_parse_hl().parse_args()
    highlight_img(args)
