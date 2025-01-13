import os
from scipy.interpolate import splprep, splev
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .segment_reader import SegmentReader
from .stroke_reader import StrokeReader
from .strokes_to_image_converter import Strokes2ImageConverter


class Parser:
    default_img_save_params = {"path": "output_img", "general_index": 0, "not_save": True}

    
    def __init__(self, image_size=(128, 128), background_image=None, padding=(0, 0), smoothing_factor=0.5, smoothing_num_points=100):
        self.image_size = image_size
        self.background_image = background_image
        self.padding = padding
        self.smoothing_factor = smoothing_factor
        self.smoothing_num_points = smoothing_num_points
        self.segment_reader = SegmentReader()
        self.stroke_reader = StrokeReader()
        self.image_converter = Strokes2ImageConverter()

    def process_data(self, segments_file, include_base_directory, data_base_directory, img_save_params):
        include_file = self.build_include_file_path(segments_file, include_base_directory, data_base_directory)

        segments = self.segment_reader.read_segment_file(segments_file)
        strokes = self.stroke_reader.read_strokes_from_include_file(include_file, segments)

        images = []
        for i, strokes_for_character in enumerate(strokes):
            img = self.image_converter.strokes_to_image(
                strokes_for_character, 
                self.image_size, 
                background_image=self.background_image,
                padding=self.padding, 
                smoothing_factor=self.smoothing_factor, 
                smoothing_num_points=self.smoothing_num_points
            )
            images.append(img)

            if 'not_save' not in img_save_params:
                img_name = self.build_img_name(img_save_params, segments[i].character_code)
                img.save(img_name)

        return images

    @staticmethod
    def build_include_file_path(segments_file, include_base_directory, data_base_directory):
        relative_path = os.path.relpath(segments_file, data_base_directory)
        parts = relative_path.split(os.sep)
        contributor = parts[0]
        return os.path.join(include_base_directory, contributor, 'data', *parts[1:])

    @staticmethod
    def build_img_name(img_save_params, character):
        if img_save_params is None:
            return f"{int(character)}.png"

        path = img_save_params.get("path", "")

        if "general_index" in img_save_params:
            index = img_save_params["general_index"]
            name = f"{int(character)}_{index}.png"
            img_save_params["general_index"] += 1
            return os.path.join(path, name)

        raise ValueError("Invalid img_save_params params.")
        
    def run_process_on_all_files_singlethread(self, 
                                              data_base_directory, 
                                              include_base_directory,
                                              img_save_params=default_img_save_params):
        images = []
        for root, _, files in os.walk(data_base_directory):
            for file in files:
                file_path = os.path.join(root, file)
                images += self.process_data(file_path, 
                                            include_base_directory, 
                                            data_base_directory,
                                            img_save_params)
        return images
    

    def run_process_on_all_files(self, 
                                 data_base_directory, 
                                 include_base_directory, 
                                 image_size=(128, 128), 
                                 background_image=None,
                                 img_save_params=default_img_save_params,
                                 num_threads=4):
        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(data_base_directory)
            for file in files
        ]

        images = []
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            process_func = partial(
                self.process_data,
                include_base_directory=include_base_directory,
                data_base_directory=data_base_directory,
                img_save_params=img_save_params
            )

            results = executor.map(process_func, file_paths)
            for result in results:
                images.extend(result)

        return images
