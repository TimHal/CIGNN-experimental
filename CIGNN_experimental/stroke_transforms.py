import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline

# Shortening
def shorten(strokes: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """
    Shorten each stroke of the prototype by one point of each end

    Args:
        strokes: List of strokes, where each stroke is a list of (x, y) tuples.
    
    Returns:
        shortened_prototype: Modified prototype with points cut off.
    """


    shortened_strokes = []
    for stroke in strokes:
        if len(stroke) < 3:
            shortened_strokes.append(stroke)
            continue
        shortened_strokes.append(stroke[1:-1])

    return shortened_strokes

def random_move(strokes: list[list[tuple[float, float]]], max_dist=(10, 10)) -> list[list[tuple[float, float]]]:
    """
    Randomly moves the points of the prototype by within the set limits (Gaussian distribution)

    Args:
        prototype: List of strokes, where each stroke is a list of (x, y) tuples.
        max_dist: 2-tuple of maximum allowed move distance
    
    Returns:
        randomly_moved_prototype: Modified prototype with points cut off.
    """    
    std_x, std_y = max_dist
    randomly_moved_strokes = []

    for stroke in strokes:
        moved_stroke = []
        for x, y in stroke:
            # Generate random displacements from a Gaussian distribution
            dx = np.random.normal(0, std_x)
            dy = np.random.normal(0, std_y)
            # Apply the displacements to the original points
            new_x = x + dx
            new_y = y + dy
            moved_stroke.append((new_x, new_y))
        randomly_moved_strokes.append(moved_stroke)

    return randomly_moved_strokes


def random_move_stroke(strokes: list[list[tuple[float, float]]], max_dist=(10, 10)) -> list[list[tuple[float, float]]]:
    """
    Randomly moves each stroke in the prototype by a consistent offset within the set limits.

    Args:
        prototype: List of strokes, where each stroke is a list of (x, y) tuples.
        max_dist: 2-tuple representing the maximum allowed move distance in x and y directions.
    
    Returns:
        randomly_moved_prototype: Modified prototype with each stroke moved by a random offset.
    """
    max_x, max_y = max_dist
    randomly_moved_strokes = []

    for stroke in strokes:
        # Generate random offsets for the entire stroke
        offset_x = np.random.uniform(-max_x, max_x)
        offset_y = np.random.uniform(-max_y, max_y)
        offset = np.array([offset_x, offset_y])
        
        # Convert stroke to a NumPy array for vectorized operations
        stroke_array = np.array(stroke)
        
        # Apply the offset to all points in the stroke
        moved_stroke = stroke_array + offset
        
        # Append the moved stroke to the result list
        randomly_moved_strokes.append(moved_stroke.tolist())

    return randomly_moved_strokes
    
# Prolonging
# For each stroke of the input, fit a cubic spline and extend it
def prolong(strokes: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """
    Extend each stroke by adding one extrapolated point before the first point
    and one after the last point, with the extension distance based on the
    average distance between consecutive points in the stroke.
    
    Args:
        prototype: List of strokes, where each stroke is a list of (x, y) tuples.
    
    Returns:
        prolonged_prototype: Modified prototype with extended strokes.
    """
    prolonged_strokes = []
    
    for stroke in strokes:
        if len(stroke) < 2:
            prolonged_strokes.append(stroke)
            continue  # Not enough points to fit a spline

        points = np.array(stroke)
        x = points[:, 0]
        y = points[:, 1]

        # Calculate differences between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Compute distances between consecutive points
        distances = np.hypot(dx, dy)
        
        # Calculate the average distance
        avg_distance = np.mean(distances) * 0.5
        
        # Parameter t based on cumulative distance
        t = np.insert(np.cumsum(distances), 0, 0)

        # Remove duplicate t values to satisfy the CubicSpline requirement
        t, unique_idx = np.unique(t, return_index=True)
        x = x[unique_idx]
        y = y[unique_idx]
        
        # Fit cubic splines
        cs_x = CubicSpline(t, x, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='natural')
        
        # Extrapolate one point before the first point
        t_before = t[0] - avg_distance
        x_before = cs_x(t_before)
        y_before = cs_y(t_before)
        
        # Extrapolate one point after the last point
        t_after = t[-1] + avg_distance
        x_after = cs_x(t_after)
        y_after = cs_y(t_after)
        
        # Construct the extended stroke
        extended_stroke = [(x_before, y_before)] + stroke + [(x_after, y_after)]
        prolonged_strokes.append(extended_stroke)
    
    return prolonged_strokes
    
# Merging
# Merge two or more points. The closer the points, the higher the change to be merged
def merge_points(strokes: list[list[tuple[float, float]]], distance_threshold=10) -> list[list[tuple[float, float]]]:
    """
    Merge points in a stroke that are closer than the specified distance.
    
    Args:
        stroke: List of (x, y) tuples representing the stroke.
        distance_threshold: Distance below which points will be merged.
    
    Returns:
        merged_stroke: List of (x, y) tuples representing the merged stroke.
    """

    merged_points_stroke = []
    for stroke in strokes:
            
        if len(stroke) < 2:
            merged_points_stroke.append(stroke)  # Not enough points to merge
        
        merged_stroke = [stroke[0]]
        for idx, point in enumerate(stroke[1:]):
            prev_point = merged_stroke[-1]
            distance = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)
            if distance < distance_threshold:
                continue
            else:
                merged_stroke.append(point)
        
        merged_points_stroke.append(merged_stroke)
    return merged_points_stroke
    

# Skipping
def skip_points(strokes: list[list[tuple[float, float]]], skip_interval=2) -> list[list[tuple[float, float]]]:
    """
    Skip points in each stroke based on the specified interval.
    
    Args:
        prototype: List of strokes, where each stroke is a list of (x, y) tuples.
        skip_interval: Interval at which points are skipped. For example, a value of 2 skips every other point.
    
    Returns:
        skipped_prototype: Modified prototype with points skipped.
    """
    res = []
    for stroke in strokes:
        if len(stroke) < 3:
            res.append(stroke)
            continue
        skipped_stroke = [point for idx, point in enumerate(stroke) if idx % skip_interval != 0]
        res.append(skipped_stroke)
    return res

# Stroke Merging
def merge_strokes(strokes: list[list[tuple[float, float]]], distance_threshold=100) -> list[list[tuple[float, float]]]:
    """
    Merge consecutive strokes if the distance between their endpoints is below the threshold.
    
    Args:
        prototype: List of strokes, where each stroke is a list of (x, y) tuples.
        distance_threshold: Maximum distance between endpoints to consider for merging.
    
    Returns:
        merged_prototype: Modified prototype with strokes merged.
    """
    if not strokes:
        return strokes

    merged_strokes = [strokes[0]]
    for stroke in strokes[1:]:
        last_stroke = merged_strokes[-1]
        if not last_stroke or not stroke:
            merged_strokes.append(stroke)
            continue
        # Calculate distance between the end of the last stroke and the start of the current stroke
        distance = np.sqrt((stroke[0][0] - last_stroke[-1][0])**2 + (stroke[0][1] - last_stroke[-1][1])**2)
        if distance < distance_threshold:
            # Merge strokes
            merged_strokes[-1].extend(stroke)
        else:
            merged_strokes.append(stroke)
    return merged_strokes

# Stroke Splitting
def split_strokes(strokes: list[list[tuple[float, float]]], split_interval=5) -> list[list[tuple[float, float]]]:
    """
    Split each stroke into multiple strokes based on the specified interval.
    
    Args:
        strokes: List of strokes, where each stroke is a list of (x, y) tuples.
        split_interval: Number of points after which to split the stroke.
    
    Returns:
        split_strokes: Modified prototype with strokes split.
    """
    split_stroke = []
    for stroke in strokes:
        if len(stroke) <= split_interval:
            split_stroke.append(stroke)
            continue
        for i in range(0, len(stroke), split_interval):
            split_stroke.append(stroke[i:i + split_interval])
    return split_stroke

def smooth_strokes(strokes: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """
    Smooth each stroke by fitting a cubic spline and re-sampling points.

    Args:
        prototype: List of strokes, where each stroke is a list of (x, y) tuples.
        num_points: Number of points to re-sample along each stroke for smoothing.

    Returns:
        smoothed_prototype: List of smoothed strokes.
    """
    smoothed_strokes = []

    for stroke in strokes:
        if len(stroke) < 2:
            smoothed_strokes.append(stroke)
            continue  # Not enough points to fit a spline

        points = np.array(stroke)
        x = points[:, 0]
        y = points[:, 1]

        # Parameter t based on cumulative distance
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        t = np.insert(np.cumsum(distances), 0, 0)

        # Remove duplicate t values to satisfy the CubicSpline requirement
        t, unique_idx = np.unique(t, return_index=True)
        x = x[unique_idx]
        y = y[unique_idx]

        if len(t) < 2:
            smoothed_strokes.append(stroke)
            continue  # Not enough unique points to fit a spline

        # Fit cubic splines
        cs_x = CubicSpline(t, x, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='natural')

        # Re-sample points along the spline
        t_new = np.linspace(t[0], t[-1], len(stroke))
        x_smooth = cs_x(t_new)
        y_smooth = cs_y(t_new)

        smoothed_stroke = list(zip(x_smooth, y_smooth))
        smoothed_strokes.append(smoothed_stroke)

    return smoothed_strokes