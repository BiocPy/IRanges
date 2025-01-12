import numpy as np

from iranges import IRanges

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_tile():
    x = IRanges([1, 5, -2, 0, 14], width=[10, 5, 6, 12, 4])

    res = x.tile(n=2)
    assert len(res) == len(x)
    assert all(np.equal(res[0]._start, [1, 6]))
    assert all(np.equal(res[0]._width, [5, 5]))

    assert all(np.equal(res[1]._start, [5, 7]))
    assert all(np.equal(res[1]._width, [2, 3]))

    assert all(np.equal(res[2]._start, [-2, 1]))
    assert all(np.equal(res[2]._width, [3, 3]))

    assert all(np.equal(res[3]._start, [0, 6]))
    assert all(np.equal(res[3]._width, [6] * 2))

    assert all(np.equal(res[4]._start, [14, 16]))
    assert all(np.equal(res[4]._width, [2, 2]))

    res = x.tile(width=3)
    assert len(res) == len(x)
    assert all(np.equal(res[0]._start, [1, 3, 6, 8]))
    assert all(np.equal(res[0]._width, [2, 3, 2, 3]))

    assert all(np.equal(res[1]._start, [5, 7]))
    assert all(np.equal(res[1]._width, [2, 3]))

    assert all(np.equal(res[2]._start, [-2, 1]))
    assert all(np.equal(res[2]._width, [3, 3]))

    assert all(np.equal(res[3]._start, [0, 3, 6, 9]))
    assert all(np.equal(res[3]._width, [3] * 4))

    assert all(np.equal(res[4]._start, [14, 16]))
    assert all(np.equal(res[4]._width, [2, 2]))


def test_sliding_windows():
    x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])

    res = x.sliding_windows(width=3)
    assert len(res) == len(x)

    assert all(np.equal(res[0]._start, range(1, 9)))
    assert all(np.equal(res[0]._width, [3] * 8))

    assert all(np.equal(res[1]._start, [5, 6, 7]))
    assert all(np.equal(res[1]._width, [3, 3, 3]))

    assert all(np.equal(res[2]._start, [-2, -1, 0, 1]))
    assert all(np.equal(res[2]._width, [3, 3, 3, 3]))
