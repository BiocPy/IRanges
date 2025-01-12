from typing import Optional, Tuple, Union

import numpy as np

__author__ = "Jayaram Kancherla"
__copyright__ = "jkanche"
__license__ = "MIT"


def normalize_array(
    x: Optional[Union[int, float, np.number, np.ndarray]], length: int, dtype: np.dtype = np.int32
) -> np.ma.MaskedArray:
    """Normalize input to masked array with proper length and type.

    Args:
        x:
            Input value (scalar, array, or None).

        length:
            Expected length for output array.

        dtype:
            Expected numpy dtype.

    Returns:
        Normalized masked array.
    """
    # If None, return a masked array of the expected length
    # so the downstream code is less complicated
    if x is None:
        return np.ma.masked_array(np.zeros(length, dtype=dtype), mask=True)

    # scalars get converted into array of length n
    if np.isscalar(x):
        return np.ma.masked_array([x] * length, dtype=dtype, mask=False)

    # everything else becomes a ndarray
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=dtype)

    # Handle length mismatch
    if len(x) < length:
        # recycle values
        repeats = length // len(x) + (1 if length % len(x) else 0)
        x = np.tile(x, repeats)[:length]
    elif len(x) > length:
        raise Exception(f"input length {len(x)} exceeds expected length {length}")

    return np.ma.masked_array(x, dtype=dtype, mask=False)


def handle_negative_coords(coords: np.ma.MaskedArray, ref_len: np.ndarray) -> np.ma.MaskedArray:
    """Convert negative coordinates to positive using reference length.

    Args:
        coords:
            Coordinate array (can have negative values).

        ref_len:
            Reference lengths for conversion.

    Returns:
        Array with negative coordinates converted to positive.
    """
    return np.ma.where((~coords.mask) & (coords < 0), ref_len + coords + 1, coords)


def clip_ranges(
    starts: np.ndarray, widths: np.ndarray, min_val: Optional[int] = None, max_val: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip ranges to specified bounds.

    Args:
        starts:
            Start positions.

        widths:
            Widths.

        min_val:
            Minimum allowed position.
            Defaults to None for no lower bound.

        max_val:
            Maximum allowed position.
            Defaults to None for no upper bound.

    Returns:
        Tuple of clipped (starts, widths) ranges.
    """
    ends = starts + widths - 1

    if min_val is not None:
        starts = np.maximum(starts, min_val)
    if max_val is not None:
        ends = np.minimum(ends, max_val)

    new_widths = np.maximum(ends - starts + 1, 0)
    return starts, new_widths


def compute_up_down(
    starts: np.ndarray,
    widths: np.ndarray,
    upstream: Union[int, np.ndarray],
    downstream: Union[int, np.ndarray],
    site: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper for promoters/terminators."""
    length = len(starts)

    if len(starts) == 0:
        return starts, widths

    upstream_arr = normalize_array(upstream, length)
    downstream_arr = normalize_array(downstream, length)

    if upstream_arr.mask.any() or downstream_arr.mask.any():
        raise ValueError("'upstream' and 'downstream' cannot be NA")

    if np.any(upstream_arr < 0) or np.any(downstream_arr < 0):
        raise ValueError("'upstream' and 'downstream' must be >= 0")

    site_pos = starts if site == "TSS" else starts + widths - 1

    new_starts = site_pos - upstream_arr.data
    new_ends = site_pos + downstream_arr.data - 1
    new_widths = new_ends - new_starts + 1

    return new_starts, new_widths


def calc_gap_and_overlap(start1: int, width1: int, start2: int, width2: int):
    """Calculate gap, overlap and relative position between two intervals."""

    end1 = start1 + width1 - 1
    end2 = start2 + width2 - 1

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap = max(0, overlap_end - overlap_start + 1)

    if end1 < start2:
        # First interval precedes second
        gap = start2 - end1 - 1
        position = "start"
    elif end2 < start1:
        # First interval follows second
        gap = start1 - end2 - 1
        position = "end"
    else:
        # Intervals overlap
        gap = -overlap
        position = "overlap"

    return gap, overlap, position


def find_interval(x: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Python implementation of R's findInterval function.

    Args:
        x:
            Values to find intervals for.

        vec:
            Sorted vector to find intervals in.

    Returns:
        NumPy array of indices indicating which interval each x value falls into.
    """
    if len(vec) == 0:
        return np.zeros(len(x), dtype=np.int32)

    indices = np.searchsorted(vec, x, side="right") - 1
    return indices
