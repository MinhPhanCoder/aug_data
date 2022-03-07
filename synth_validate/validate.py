import warnings
import argparse
from pathlib import Path


class Validate:
    allowed_output_types = ['.png', '.jpg', '.jpeg']
    allowed_background_types = ['.png', '.jpg', '.jpeg']

    def _validate_and_process_args(self, args: argparse):
        self.silent = args.silent
        # Check count
        assert args.count > 0, "'count' must be greater than 0"
        self.count = args.count
        # Check format image
        if args.output_type is None:
            self.output_type = '.jpg'
        else:
            if args.output_type[0] != '.':
                self.output_type = f'.{args.output_type}'
            assert self.output_type in self.allowed_output_types, f'Unsupported format : {self.output_type}'
        # Check the Input and Output directory
        self._validate_and_process_output_directory(args)
        self._validate_and_process_input_directory(args)

    def _validate_and_process_input_directory(self, args: argparse):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'Input path does not exist: {args.input_dir}'
        for x in self.input_dir.iterdir():
            if x.name == 'foregrounds':
                self.foregrounds_dir = x
            elif x.name == 'backgrounds':
                self.backgrounds_dir = x
        assert self.foregrounds_dir is not None, 'Foregrounds folder is not found in the Input folder'
        assert self.backgrounds_dir is not None, 'Backgrounds folder is not found in the Input folder'
        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_output_directory(self, args: argparse):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'images'
        self.masks_output_dir = self.output_dir / 'masks'
        self.highlight_output_dir = self.output_dir / 'highlight'
        # Create directory if does not exist
        self.output_dir.mkdir(exist_ok=True)
        self.images_output_dir.mkdir(exist_ok=True)
        self.masks_output_dir.mkdir(exist_ok=True)
        self.highlight_output_dir.mkdir(exist_ok=True)
        if not self.silent:
            # Check if there are any images in the output/images folder
            for _ in self.images_output_dir.iterdir():
                # overwrite ?
                should_continue = input('output/image directory contains files, do you want to overwrite.\nContinue (y/n) ? ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

    def _validate_and_process_foregrounds(self):
        """
        Check the foregrounds folder structure
        ...
        - foregrounds
            - sub_category1
                - category1
                    + dog_category1.png
                - category2
                    + dog_category2.png
            - sub_category2
                - category1
                    + cat_category1.png
                - category2
                    + cat_category2.png
        """
        self.foregrounds_dict = dict()
        # Check the foreground folder
        for sub_category_dir in self.foregrounds_dir.iterdir():
            if not sub_category_dir.is_dir():
                warnings.warn(f'Unwanted file found in foregrounds/... folder, ignoring: {sub_category_dir}')
                continue
            # Check subfolders of foreground
            for category_dir in sub_category_dir.iterdir():
                if not category_dir.is_dir():
                    warnings.warn(f'Unwanted file found in foregrounds/.../... folder, ignoring: {category_dir}')
                    continue
                # Check pictures in subfolder
                for path_image_file in category_dir.iterdir():
                    if not path_image_file.is_file():
                        warnings.warn(f'Found folder in subfolder, ignoring: {str(path_image_file)}')
                        continue
                    if path_image_file.suffix != '.png':
                        warnings.warn(f'Found images in formats other than PNG, skipping: {str(path_image_file)}')
                        continue
                    # Write to dict directory structure
                    sub_category = sub_category_dir.name
                    category = category_dir.name
                    if sub_category not in self.foregrounds_dict:
                        self.foregrounds_dict[sub_category] = dict()
                    if category not in self.foregrounds_dict[sub_category]:
                        self.foregrounds_dict[sub_category][category] = []
                    self.foregrounds_dict[sub_category][category].append(path_image_file)
        assert len(self.foregrounds_dict) > 0, 'Please check your foregrounds directory structure !!!'

    def _validate_and_process_backgrounds(self):
        self.backgrounds = []
        for path_image_file in self.backgrounds_dir.iterdir():
            if not path_image_file.is_file():
                warnings.warn(f'Unwanted folder found in backgrounds . folder, ignoring: {path_image_file}')
                continue
            if path_image_file.suffix not in self.allowed_background_types:
                warnings.warn(f'background must have the format --> {str(self.allowed_background_types)}, ignoring: {path_image_file}')
                continue
            self.backgrounds.append(path_image_file)
        assert len(self.backgrounds) > 0, 'Please add pictures to the background folder !!!'
