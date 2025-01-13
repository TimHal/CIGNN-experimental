import re

class Segment:
    def __init__(self, segment_start, segment_end, subsegment_start, subsegment_end, character_code):
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.subsegment_start = subsegment_start
        self.subsegment_end = subsegment_end
        self.character_code = character_code

    @classmethod
    def from_match(cls, match):
        """
        Alternative constructor to create a Segment instance from a regex match object.
        """
        segment_range = match[0]
        character = match[2]
        sub_start, sub_end = None, None

        if '-' in segment_range:
            start, end = segment_range.split('-')
        else:
            start, end = segment_range, segment_range
        if len(start.split(':')) > 1:
            start, sub_start = map(int, start.split(':'))
        if len(end.split(':')) > 1:
            end, sub_end = map(int, end.split(':'))

        if len(character) == 2 and character[0] == '\\':
            character = character[1]

        return cls(
            segment_start=int(start),
            segment_end=int(end) if end else int(start),
            subsegment_start=sub_start,
            subsegment_end=sub_end,
            character_code=ord(character),
        )

    def __repr__(self):
        return (f"Segment(start={self.segment_start}, end={self.segment_end}, "
                f"sub_start={self.subsegment_start}, sub_end={self.subsegment_end}, "
                f"character_code={chr(self.character_code)})")
    

class SegmentReader:
    segment_pattern = re.compile(r'\.SEGMENT CHARACTER (\d+(?::\d+)?(-\d+(?::\d+)?)?) (?:\?|OK)\s+\"(.+)\"')

    def read_segment_file(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        matches = self.segment_pattern.findall(content)
        segments = [Segment.from_match(match) for match in matches]
        return segments