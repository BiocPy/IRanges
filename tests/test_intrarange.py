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

def test_narrow():
    starts = [1, 20, 25, 33]
    widths = [19, 5, 8, 5]
    x = IRanges(starts, widths)

    res = x.narrow(start=4)
    assert all(np.equal(res.start, [4, 23, 28, 36]))
    assert all(np.equal(res.width, [16, 2, 5, 2]))

    res = x.narrow(start = 4, width = 2)
    assert all(np.equal(res.start, [4, 23, 28, 36]))
    assert all(np.equal(res.width, [2, 2, 2, 2]))

    res = x.narrow(start = -4, width = 2)
    assert all(np.equal(res.start, [16, 21, 29, 34]))
    assert all(np.equal(res.width, [2, 2, 2, 2]))

    res = x.narrow(start=4, end = -2)
    assert all(np.equal(res.start, [4, 23, 28, 36]))
    assert all(np.equal(res.width, [15, 1, 4, 1]))
    
def test_resize():
    starts = [1, 20, 25, 33]
    widths = [19, 5, 8, 5]
    x = IRanges(starts, widths)

    res = x.resize(200)
    assert all(np.equal(res.start, starts))
    assert all(np.equal(res.width, [200] * 4))


    res = x.resize(2, fix="end")
    assert all(np.equal(res.start, [18, 23, 31, 36]))
    assert all(np.equal(res.width, [2] * 4))  