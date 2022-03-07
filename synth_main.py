import os
import random
import numpy as np
import synth_config
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import Polygon
from synth_validate.validate import Validate
from synth_utils.utils_load_data import CustomDataset
from synth_utils import utils_synth, utils_write_label
from synth_aug.aug import aug_foreground, aug_background
from typing import List, Tuple, Dict, Union


class ImageComposition(Validate):
    def __init__(self):
        self._validate_and_process_args(args)
        self.width = 1000
        self.height = 1000
        self.zero_padding = 8  # 00000027.png
        self.max_foregrounds = synth_config.max_foregrounds
        self.mask_colors = [(255, 255, 255)]
        if not synth_config.paste_all_position:
            self.data_position_backgrounds = CustomDataset()
            self.data_position_backgrounds.load_dataset(self.input_dir / "backgrounds")
            self.data_position_backgrounds.prepare()

    def generate_images(self):
        print(f'----- Generating {self.count} images with masks -----')
        dict_full_polygon_op = dict()
        for i in tqdm(range(self.count)):
            path_background = utils_synth.random_from_list(self.backgrounds)
            num_foregrounds = random.randint(1, self.max_foregrounds)
            lst_path_foregrounds = self.random_foregrounds(num_foregrounds)
            img_composite, img_mask, ls_all_polygon = self._compose_images(lst_path_foregrounds, path_background)
            save_filename = f'{i:0{self.zero_padding}}'
            self.save_img_generator(img_composite, img_mask, save_filename)
            dict_full_polygon_op[save_filename + self.output_type] = ls_all_polygon
        utils_write_label.export_json_output_custom(dict_full_polygon_op, self.output_dir / 'images')
        utils_synth.write_pkl(Path(self.output_dir) / 'masks' / 'mask_definitions.pkl', dict_full_polygon_op)

    def _compose_images(self, lst_path_foregrounds: List[str], path_background: str) -> Tuple[object, object, List[List]]:
        # Open background and convert to RGBA
        img_background = aug_background(path_background)
        composite_mask = np.array(Image.new('L', img_background.size, 0))
        lst_polygon_bg: List[Polygon]
        if not synth_config.paste_all_position:
            lst_polygon_bg = self.position_enable_paste(path_background)
        else:
            lst_polygon_bg = [Polygon([(0, 0),
                                      (0, img_background.size[1]),
                                      (img_background.size[0], img_background.size[1]),
                                      (img_background.size[0], 0)])]
        ls_pasted = []
        ls_all_polygon = []
        for foreground in lst_path_foregrounds:
            chk_pasted = False
            while not chk_pasted:
                # Aug foreground
                img_foreground = aug_foreground(foreground['foreground_path'])
                paste_position = utils_synth.random_x_y_paste(img_background, img_foreground)
                polygon_paste = self.get_polygon_paste(img_background, img_foreground, paste_position)
                if polygon_paste is None:
                    continue
                chk_pasted = self.chk_enable_paste(polygon_paste, lst_polygon_bg, ls_pasted)
            img_background, new_alpha_mask = self.paste_foreground_to_background(img_background, img_foreground, paste_position)
            composite_mask += np.array(new_alpha_mask)
            ls_all_polygon.append([polygon_paste, foreground['sub_category'], foreground['category']])
        composite_mask = Image.fromarray(composite_mask, 'L')
        return img_background, composite_mask, ls_all_polygon

    def random_foregrounds(self, num_foregrounds: int) -> List[Dict]:
        lst_path_foregrounds = []
        for _ in range(num_foregrounds):
            sub_category, category, path_foreground = utils_synth.random_foreground(self.foregrounds_dict)
            lst_path_foregrounds.append({
                'sub_category': sub_category,
                'category': category,
                'foreground_path': path_foreground
            })
        return lst_path_foregrounds

    def position_enable_paste(self, path_background: str) -> List[Polygon]:
        lst_position_enable_paste = self.data_position_backgrounds.get_polygon_with_name(os.path.basename(path_background))
        return utils_synth.convert2polygon(lst_position_enable_paste)

    def save_img_generator(self, img_composite: object, img_mask: object, save_filename: str):
        path_composite = self.output_dir / 'images' / f'{save_filename}{self.output_type}'
        # Remove alpha channel
        img_composite = img_composite.convert('RGB')
        img_composite.save(path_composite)
        # Mask always .png to avoid quality loss
        path_mask = self.output_dir / 'masks' / f'{save_filename}.png'
        img_mask.save(path_mask)

    @staticmethod
    def get_polygon_paste(img_background: object, img_foreground: object, paste_position: Tuple[int, int, int, int]) -> Union[None, Polygon]:
        alpha_mask = img_foreground.getchannel(3)
        background_mask = Image.new('L', img_background.size, color=0)
        background_mask.paste(alpha_mask, paste_position)
        polygon_paste = utils_synth.get_polygon_findcontour(np.array(background_mask))
        if polygon_paste is None:
            return None
        return polygon_paste

    @staticmethod
    def chk_enable_paste(polygon_paste: Polygon, ls_polygon_bg: List[Polygon], ls_pasted: List[Polygon]) -> bool:
        if not utils_synth.check_overlap_in_lstpolygon(polygon_paste, ls_pasted):
            for polygon_bg in ls_polygon_bg:
                if polygon_bg.contains(polygon_paste):
                    ls_pasted.append(polygon_paste)
                    return True
        return False

    @staticmethod
    def paste_foreground_to_background(img_background: object, img_foreground: object, paste_position: Tuple[int, int, int, int]) -> Tuple[object, object]:
        new_foreground_image = Image.new('RGBA', img_background.size, color=(0, 0, 0, 0))
        new_foreground_image.paste(img_foreground, paste_position)
        alpha_mask = img_foreground.getchannel(3)
        new_alpha_mask = Image.new('L', img_background.size, color=0)
        new_alpha_mask.paste(alpha_mask, paste_position)
        img_background = Image.composite(new_foreground_image, img_background, new_alpha_mask)
        return img_background, new_alpha_mask


if __name__ == "__main__":
    args = utils_synth.get_parser().parse_args()
    image_comp = ImageComposition()
    image_comp.generate_images()


