from PIL import Image, ImageDraw
import numpy as np
from scipy.interpolate import splprep, splev

class Strokes2ImageConverter:
    @staticmethod
    def normalize_coords_with_padding(coords, mins, maxs, size, padding):
        ranges = [maxs[i] - mins[i] for i in range(2)]
        effective_sizes = [size[i] - 2 * padding[i] for i in range(2)]

        scaling_factor = (
            1 if ranges[0] == 0 and ranges[1] == 0 else
            effective_sizes[0] / ranges[0] if ranges[1] == 0 else
            effective_sizes[1] / ranges[1] if ranges[0] == 0 else
            min(effective_sizes[0] / ranges[0], effective_sizes[1] / ranges[1])
        )

        offsets = [
            padding[i] + (effective_sizes[i] - ranges[i] * scaling_factor) / 2
            for i in range(2)
        ]

        normalized_x = (coords[0] - mins[0]) * scaling_factor + offsets[0]
        normalized_y = size[1] - ((coords[1] - mins[1]) * scaling_factor + offsets[1])
        return normalized_x, normalized_y

    @staticmethod
    def normalize_stroke(stroke, mins, maxs, image_size, padding):
        return [
            Strokes2ImageConverter.normalize_coords_with_padding(coords, mins, maxs, image_size, padding)
            for coords in stroke
        ]

    @staticmethod
    def filter_duplicate_points(points):
        seen = set()
        unique_points = []
        for point in points:
            if point not in seen:
                seen.add(point)
                unique_points.append(point)
        return unique_points

    @staticmethod
    def smooth_stroke(stroke, smoothing_factor=0.5, num_points=100):
        if len(stroke) < 4:
            return stroke

        unique_points = Strokes2ImageConverter.filter_duplicate_points(stroke)
        if len(unique_points) < 4:
            return stroke

        x, y = zip(*unique_points)
        tck, _ = splprep([x, y], s=smoothing_factor)
        smoothed_points = splev(np.linspace(0, 1, num_points), tck)
        return list(zip(smoothed_points[0], smoothed_points[1]))

    def strokes_to_image(self, 
                         strokes, 
                         image_size=(128, 128), 
                         stroke_width=3, 
                         background_image=None, 
                         padding=(0, 0), 
                         smoothing_factor=0.5, 
                         smoothing_num_points=100):
        if not strokes:
            return Image.new('1', image_size, 1)

        mins, maxs = self.find_mins_maxs(strokes)
        img = background_image or Image.new('1', image_size, 1)
        draw = ImageDraw.Draw(img)

        for stroke in strokes:
            if not stroke:
                continue

            stroke = self.normalize_stroke(stroke, mins, maxs, image_size, padding)
            if smoothing_factor:
                stroke = self.smooth_stroke(stroke, smoothing_factor * len(stroke), smoothing_num_points)

            for i in range(len(stroke) - 1):
                draw.line((*stroke[i], *stroke[i + 1]), fill=0, width=stroke_width)
                
            if len(stroke) == 1:
                draw.point(stroke[0], fill=0)

        return img

    @staticmethod
    def find_mins_maxs(strokes):
        all_coords = [coord for stroke in strokes for coord in stroke]
        return (
            (min(coord[0] for coord in all_coords), min(coord[1] for coord in all_coords)),
            (max(coord[0] for coord in all_coords), max(coord[1] for coord in all_coords))
        )
