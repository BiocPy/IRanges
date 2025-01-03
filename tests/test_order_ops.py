import numpy as np
import pytest
from iranges import IRanges

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_sort():
    starts = [-2, 6, 9, -4, 1, 0, -6, 3, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 0, 3]
    x = IRanges(starts, widths)

    sorted = x.sort()
    assert all(np.equal(sorted.start, [-6, -4, -2, 0, 1, 3, 6, 9, 10]))
    assert all(np.equal(sorted.width, [2, 1, 5, 3, 4, 0, 0, 6, 3]))


def test_sort_decreasing():
    starts = [-2, 6, 9, -4, 1, 0, -6, 3, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 0, 3]
    x = IRanges(starts, widths)

    sorted_iranges = x.sort(decreasing=True)
    assert all(np.equal(sorted_iranges.start, [-6, -4, -2, 0, 1, 3, 6, 9, 10][::-1]))
    assert all(np.equal(sorted_iranges.width, [2, 1, 5, 3, 4, 0, 0, 6, 3][::-1]))


def test_order():
    starts = [-2, 6, 9, -4, 1, 0, -6, 3, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 0, 3]
    x = IRanges(starts, widths)

    order = x.order()
    assert all(np.equal(order, [6, 3, 0, 5, 4, 7, 1, 2, 8]))
