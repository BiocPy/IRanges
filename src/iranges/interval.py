from typing import List, Optional, Tuple, Union

import numpy as np

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def create_np_interval_vector(
    intervals: "IRanges",
    with_reverse_map: bool = False,
    force_size: Optional[int] = None,
    dont_sum: bool = False,
    value: Union[int, float] = 1,
) -> Tuple[np.ndarray, Optional[List]]:
    """Represent intervals and calculate coverage.

    Args:
        intervals (IRanges): Input intervals.
        with_reverse_map (bool, optional): Return map of indices? Defaults to False.
        force_size (int, optional): Force size of the array.
        dont_sum (bool, optional): Do not sum. Defaults to False.
        value (Union[int, float], optional): Default value to increment. Defaults to 1.

    Returns:
        Tuple[ndarray, Optional[List]]: A numpy array representing
        coverage from the intervals and optionally the index map.
    """
    if len(intervals) == 0:
        return np.zeros(0), None

    max_end = force_size
    if max_end is None:
        max_end = max(intervals.get_end())
    else:
        max_end += 1

    _type = np.int32
    if isinstance(value, float):
        _type = np.float32

    cov = np.zeros(max_end, dtype=_type)

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


def calc_gap_and_overlap(
    first: Tuple[int, int], second: Tuple[int, int]
) -> Tuple[Optional[int], Optional[int]]:
    """Calculate gap and/or overlap between two intervals.

    Args:
        first (Tuple): Interval containing start and end positions. `end` is non-inclusive.
        second (Tuple): Interval containing start and end positions. `end` is non-inclusive.
    """
    _gap = None
    _overlap = None

    if first[0] < second[1] and first[1] > second[0]:
        _overlap = min(first[1], second[1]) - max(first[0], second[0])
    else:
        _gap = None
        if second[0] >= first[1]:
            _gap = second[0] - first[1]
        elif first[0] >= second[1]:
            _gap = first[0] - second[1]

    return (_gap, _overlap)


def solve_interval_args(
    start: Optional[Union[int, List[int], np.ndarray]] = None,
    end: Optional[Union[int, List[int], np.ndarray]] = None,
    width: Optional[Union[int, List[int], np.ndarray]] = None,
) -> "IRanges":
    """Solve for interval arguments.

    Args:
        start:
            Start position. Defaults to None.

        end:
            End position. Defaults to None.

        width:
            Width. Defaults to None.

    Returns:
        An ``IRanges`` object containing the solved interval.
    """

    print("incoming", start, width, end)
    _start = None
    _width = None

    if (all(x is not None for x in (start, end, width))) or (
        all(x is None for x in (start, end, width))
    ):
        raise ValueError(
            "Two out of three ('start', 'end' or 'width') arguments must be provided."
        )

    print("WIDTHHHHH::", width)
    if width is not None:
        if (isinstance(width, int) and width < 0) or (
            isinstance(width, np.ndarray) and any(x < 0 for x in width)
        ):
            raise ValueError("'width' cannot be negative.")

        if start is None and end is None:
            raise ValueError(
                "If 'width' is provided, either 'start' or 'end' must be provided."
            )

    if start is not None:
        _start = start

        if width is not None:
            _width = width
        elif end is not None:
            _width = end - start
    elif end is not None:
        if start is not None:
            _start = start
            _width = end - start
        elif width is not None:
            _width = width
            _start = end - width
    elif width is not None:
        _width = width

        if start is not None:
            _start = start
        elif end is not None:
            _start = end - width

    print("in solving", _start, _width)

    # if (isinstance(_width, int) and _width < 0) or (
    #     isinstance(_width, np.ndarray) and any(x < 0 for x in _width)
    # ):
    #     raise ValueError(
    #         "Negative values not allowed for 'width'. Failed solving for provided arguments."
    #     )

    return _start, _width
