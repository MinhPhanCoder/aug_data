import argparse
import numpy as np
from pathlib import Path
from synth_utils import utils_visualize
from synth_utils.utils_load_data import CustomDataset


def get_parse_hl():
    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True,
                        help="Folder containing photos to highlight")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True,
                        help="Folder containing photos after highlighting")
    parser.add_argument("--count", type=int, dest="count", default=10,
                        help="Amount to highlight")
    return parser


def extract_bboxes(mask):
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def highlight_img(args):
    CustomDS = CustomDataset()
    CustomDS.load_dataset(Path(args.input_dir))
    CustomDS.prepare()
    for image_id in CustomDS.image_ids[0:args.count]:
        image = CustomDS.load_image(image_id)
        mask, class_ids = CustomDS.load_mask(image_id)
        bbox = extract_bboxes(mask)
        print("image_id ", image_id, CustomDS.image_reference(image_id))
        fn = str(Path(args.output_dir) / CustomDS.image_info[image_id]["id"])
        utils_visualize.display_instances(image, bbox, mask, class_ids, CustomDS.class_names, path_save=fn)


if __name__ == "__main__":
    args = get_parse_hl().parse_args()
    highlight_img(args)
