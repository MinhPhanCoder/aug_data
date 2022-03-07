import random
import pickle
import argparse
import warnings
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
from typing import List, Dict, Any, Tuple


def get_parser() -> argparse:
    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True, help="The input directory. \
                           This contains a 'backgrounds' directory of pngs or jpgs, and a 'foregrounds' directory which \
                           contains subcategory directories (e.g. 'animal', 'vehicle'), each of which contain category \
                           directories (e.g. 'horse', 'bear'). Each category directory contains png images of that item on a \
                           transparent background (e.g. a grizzly bear on a transparent background).")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help="The directory where images, masks, \
                           and json files will be placed")
    parser.add_argument("--count", type=int, dest="count", required=True,
                        help="number of composed images to create")
    parser.add_argument("--output_type", type=str, dest="output_type", help="png or jpg (default)")
    parser.add_argument("--silent", action='store_true', help="silent mode; doesn't prompt the user for input, \
                           automatically overwrites files")
    return parser


def show_wait_destroy(win_name: str, img: object):
    cv.namedWindow(win_name, 0)
    cv.imshow(win_name, img)
    cv.waitKey(0)


def write_pkl(path: str, data: Any):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pkl(path: str) -> object:
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data


def random_from_list(lst_obj: List) -> Any:
    return random.choice(lst_obj)


def random_foreground(foregrounds_dict: Dict) -> Tuple[str, str, str]:
    sub_category = random.choice(list(foregrounds_dict.keys()))
    category = random.choice(list(foregrounds_dict[sub_category].keys()))
    foreground_path = random.choice(foregrounds_dict[sub_category][category])
    return sub_category, category, foreground_path


def random_x_y_paste(background_img: object, foreground_img:object) -> Tuple[int, int, int, int]:
    max_x_position = background_img.size[0] - foreground_img.size[0]
    max_y_position = background_img.size[1] - foreground_img.size[1]
    assert max_x_position >= 0 and max_y_position >= 0, "size foreground > size background"
    paste_position = (random.randint(0, max_x_position), random.randint(0, max_y_position))
    return paste_position


def convert2polygon(lst_position_enable_paste: List[Dict]) -> List[Polygon]:
    ls_polygon = []
    for ele in lst_position_enable_paste:
        ls_x_y = list(zip(ele['all_points_x'], ele['all_points_y']))
        ls_polygon.append(Polygon(ls_x_y))
    return ls_polygon


def is_overlap_polygon(p1: Polygon, p2: Polygon) -> bool:
    # Check overlap polygon coordinate
    return p1.intersects(p2)


def check_overlap_in_lstpolygon(p1: Polygon, ls_polygon: List[Polygon]) -> bool:
    # Check a polygon with list polygon
    for polygon in ls_polygon:
        if is_overlap_polygon(p1, polygon):
            return True
    return False


def get_polygon_findcontour(image: object) -> Polygon:
    alpha_threshold = 200
    kernel = np.ones((3, 3), np.uint8)
    image = cv.dilate(image, kernel)
    image = np.array(np.greater(np.array(image), alpha_threshold), dtype=np.uint8)
    image = np.uint8(image)
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        warnings.warn("Found 2 contour / 1 object")
        return None
    max_contour = [sorted(contours, key=cv.contourArea, reverse=True)[0]]
    for ind, values in enumerate(max_contour):
        ls_x = [val[0][0] for val in values]
        ls_y = [val[0][1] for val in values]
    return Polygon(list(zip(ls_x, ls_y)))
