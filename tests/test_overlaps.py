from iranges import IRanges
import numpy as np

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_find_overlaps():
    query = IRanges([1, 4, 9], [5, 4, 2])
    subject = IRanges([2, 2, 10], [1, 2, 3])

    res = query.find_overlaps(subject)
    assert res == [[0], [0], [2]]

    res = query.find_overlaps(subject, max_gap=0)
    assert res == [[0], [0, 1], [2]]

def test_subset_overlaps():
    query = IRanges([1,4], [5, 4])
    subject = IRanges([2, 2], [1, 2])

    res = query.subset_by_overlaps(subject)
    assert all(np.equal(res.start, [1]))
    assert all(np.equal(res.width, [5]))