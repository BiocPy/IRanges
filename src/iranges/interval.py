from typing import List, Optional, Tuple

from numpy import ndarray, zeros

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def create_np_interval_vector(
    intervals: "IRanges",
    with_reverse_map: bool = False,
    force_size: Optional[int] = None,
    dont_sum: bool = False,
    value: int = 1,
) -> Tuple[ndarray, Optional[List]]:
    """Represent intervals and calculate coverage.

    Args:
        intervals (IRanges): Input intervals.
        with_reverse_map (bool, optional): Return map of indices? Defaults to False.
        force_size (int, optional): Force size of the array.
        dont_sum (bool, optional): Do not sum. Defaults to False.
        value (int, optional): Default value to increment. Defaults to 1.

    Returns:
        Tuple[ndarray, Optional[List]]: A numpy array representing
        coverage from the intervals and optionally the index map.
    """
    if len(intervals) == 0:
        return zeros(0), None

    max_end = force_size
    if max_end is None:
        max_end = max(intervals.get_end())

    cov = zeros(max_end)

    revmap = None
    if with_reverse_map:
        revmap = [[] for _ in range(max_end)]

    counter = 0
    for name, row in intervals:
        _start = row.start[0]
        _end = row.end[0]

        print(_start, _end)
        if dont_sum:
            cov[_start:_end] = value
        else:
            cov[_start:_end] += value

        if with_reverse_map:
            _ = [
                revmap[x].append(name if name is not None else counter + 1)
                for x in range(_start, _end)
            ]

        counter += 1
    return cov[1:], revmap
