import numpy as np
import pytest
from iranges import IRanges

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_range():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    range = x.range()
    assert all(np.equal(range.start, [-6]))
    assert all(np.equal(range.width, [21]))
    assert all(np.equal(range.end, [15]))
