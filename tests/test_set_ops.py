import numpy as np
import pytest
from iranges import IRanges

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_union():
    x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])
    y = IRanges([14, 0, -5, 6, 18], [7, 3, 8, 3, 3])

    res = x.union(y)
    assert all(np.equal(res.start, [-5, 14]))
    assert all(np.equal(res.width, [17, 7]))


def test_setdiff():
    x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])
    y = IRanges([14, 0, -5, 6, 18], [7, 3, 8, 3, 3])

    res = x.setdiff(y)
    assert all(np.equal(res.start, [3, 9]))
    assert all(np.equal(res.width, [3, 3]))


def test_intersect():
    x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])
    y = IRanges([14, 0, -5, 6, 18], [7, 3, 8, 3, 3])

    res = x.intersect(y)
    assert all(np.equal(res.start, [-2, 6, 14]))
    assert all(np.equal(res.width, [5, 3, 4]))

    res = y.intersect(x)
    assert all(np.equal(res.start, [-2, 6, 14]))
    assert all(np.equal(res.width, [5, 3, 4]))


def test_intersect_ncls():
    x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])
    y = IRanges([14, 0, -5, 6, 18], [7, 3, 8, 3, 3])

    res = x.intersect_ncls(y)
    assert all(np.equal(res.start, [-2, 6, 14]))
    assert all(np.equal(res.width, [5, 3, 4]))

    res = y.intersect_ncls(x)
    assert all(np.equal(res.start, [-2, 6, 14]))
    assert all(np.equal(res.width, [5, 3, 4]))
