import numpy as np
import pytest
from iranges import IRanges

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_shift():
    starts = [1, 20, 25, 25, 33]
    widths = [19, 5, 0, 8, 5]
    x = IRanges(starts, widths)

    sorted = x.shift(shift=-3)
    assert all(np.equal(sorted.start, [-2, 17, 22, 22, 30]))
    assert all(np.equal(sorted.width, widths))
