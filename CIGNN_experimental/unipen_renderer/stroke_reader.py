import re

class StrokeReader:
    coords_pattern = re.compile(r"\.PEN_DOWN\n((?:[\t\s]*-?\d+(?:\s+-?\d+){1,2}\n)+)(?:\.PEN_UP|\s*\n)")

    def read_strokes_from_include_file(self, file_path, segments):
        strokes = []
        with open(file_path, 'r') as file:
            content = file.read()
            
        pen_downs = list(self.coords_pattern.finditer(content))
        for segment in segments:
            relevant_pen_downs = pen_downs[segment.segment_start:segment.segment_end + 1]
            symbol_strokes = []
            for i, pen_down_match in enumerate(relevant_pen_downs):
                pen_down_content = pen_down_match.group(1)

                effective_start_slice_index = segment.subsegment_start if i == 0 else None
                effective_end_slice_index = segment.subsegment_end if (i == len(relevant_pen_downs) - 1) else None
                if effective_end_slice_index != None:
                    effective_end_slice_index += 1
                effective_slice = slice(effective_start_slice_index, effective_end_slice_index)
                symbol_stroke = [
                    tuple(map(int, line.split()[:2]))
                    for line in pen_down_content.strip().split('\n')[effective_slice]
                    if line.strip()
                ]

                symbol_strokes.append(symbol_stroke)

            strokes.append(symbol_strokes)
        return strokes