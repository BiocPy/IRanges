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


def broadcast_to_ranges(*arrays: Tuple[np.ndarray, str]) -> Tuple[np.ndarray, ...]:
    """Broadcast multiple arrays to match range length.

    Args:
        arrays:
            Tuple of (array, name) pairs.

    Returns:
        Tuple of broadcasted arrays
    """
    max_len = max(len(arr) for arr, _ in arrays)
    result = []

    for arr, name in arrays:
        if len(arr) == 1:
            result.append(np.full(max_len, arr[0], dtype=arr.dtype))
        elif len(arr) == max_len:
            result.append(arr)
        else:
            raise Exception(f"{name} length {len(arr)} cannot be broadcast to length {max_len}")

    return tuple(result)


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

def calc_gap_and_overlap(first: Tuple[int, int], second: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
    """Calculate gap and/or overlap between two intervals.

    Args:
        first:
            Interval containing start and end positions.
            `end` is non-inclusive.

        second:
            Interval containing start and end positions.
            `end` is non-inclusive.
    """
    if min(first[1], second[1]) > max(first[0], second[0]):
        _overlap = min(first[1], second[1]) - max(first[0], second[0])
        return (None, _overlap)

    _gap = None
    if second[0] >= first[1]:
        _gap = second[0] - first[1]
    elif first[0] >= second[1]:
        _gap = first[0] - second[1]

    return (_gap, None)