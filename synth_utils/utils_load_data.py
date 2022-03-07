import os
import json
import cv2 as cv
import skimage.io
import numpy as np
import skimage.draw
import skimage.color
from typing import List, Dict, Tuple


class Dataset(object):
    def __init__(self, class_map=None) -> object:
        self._image_ids: np.array() = []
        self.image_info: List[Dict] = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.num_classes: int = None
        self.class_ids: np.array() = None
        self.class_names: List[str] = None
        self.num_images: int = None
        self.class_from_source_map: Dict = None
        self.image_from_source_map: Dict = None
        self.sources: Dict = None

    def add_class(self, source: str, class_id: int, class_name: str):
        assert "." not in source, "Source name cannot contain a dot"
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source: str, image_id: int, path: str, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def prepare(self, class_map=None):
        def clean_name(name: str):
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id for info, id in zip(self.class_info,
                                                                                                         self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id for info, id in zip(self.image_info,
                                                                                                         self.image_ids)}

        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    @property
    def image_ids(self):
        return self._image_ids

    @staticmethod
    def image_reference(image_id: int) -> str:
        return ""

    def load_image(self, image_id: int) -> object:
        image = cv.imread(self.image_info[image_id]['path'])
        return image

    def load_mask(self, image_id: int) -> Tuple[object, object]:
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

    def get_polygon_with_name(self, name: str) -> List[Dict]:
        for img_inf in self.image_info:
            if img_inf["id"] == name:
                return img_inf["polygons"]


class CustomDataset(Dataset):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.source_default = "DATASET"

    def load_dataset(self, dataset_dir: str):
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [element for element in annotations if element['regions']]
        i = 1
        lst_added: List[str] = []
        for annotation in annotations:
            if type(annotation['regions']) is dict:
                polygons = [r['shape_attributes'] for r in annotation['regions'].values()]
                regions = [r['region_attributes'] for r in annotation['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in annotation['regions']]
                regions = [r['region_attributes'] for r in annotation['regions']]
            lst_class: List[str] = []
            for region in regions:
                cls_name = list(region['class'].keys())[0]
                lst_class.append(cls_name)
                if cls_name not in lst_added:
                    lst_added.append(cls_name)
                    self.add_class(self.source_default, i, cls_name)
                    i += 1
            image_path = os.path.join(dataset_dir, annotation['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                self.source_default,
                image_id=annotation['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                classname=lst_class)

    def load_mask(self, image_id: int) -> Tuple[object, object]:
        image_info = self.image_info[image_id]
        if image_info["source"] != self.source_default:
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        lst_classname = info['classname']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        class_ids = np.array([self.class_names.index(name) for name in lst_classname])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id: int) -> str:
        info = self.image_info[image_id]
        if info["source"] == self.source_default:
            return info["path"]
        super(self.__class__).image_reference(self, image_id)


if __name__ == "__main__":
    dataset = CustomDataset()
    dataset.load_dataset(r"../datatset/input/backgrounds")
    dataset.prepare()
    print(dataset.get_polygon_with_name("bg1.jpg"))

