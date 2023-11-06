from iranges import IRanges

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_find_overlaps():
    query = IRanges([1, 4, 9], [5, 4, 2])
    subject = IRanges([2, 2, 10], [1, 2, 3])

    res = query.find_overlaps(subject)
    assert res  == [[0], [0], [2]]

    res = query.find_overlaps(subject, max_gap=0)
    assert res == [[0], [0, 1], [2]]
