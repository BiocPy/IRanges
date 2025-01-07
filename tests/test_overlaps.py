from iranges import IRanges
import numpy as np

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_find_overlaps():
    query = IRanges([1, 4, 9], [5, 4, 2])
    subject = IRanges([2, 2, 10], [1, 2, 3])

    res = query.find_overlaps(subject)
    print(res)
    assert np.all(res.get_column("self_indices") == [0, 0, 2])
    assert np.all(res.get_column("query_indices") == [0, 1, 2])

    res = query.find_overlaps(subject, max_gap=0)
    assert np.all(res.get_column("self_indices") == [0, 0, 1, 2])
    assert np.all(res.get_column("query_indices") == [0, 1, 1, 2])

    res = query.find_overlaps(subject, select="first")
    assert np.all(res.get_column("self_indices") == [0, 0, 2])
    assert np.all(res.get_column("query_indices") == [0, 1, 2])

    res = query.find_overlaps(subject, select="last")
    assert np.all(res.get_column("self_indices") == [0, 0, 2])
    assert np.all(res.get_column("query_indices") == [0, 1, 2])

    res = query.find_overlaps(subject, select="arbitrary")
    assert np.all(res.get_column("self_indices") == [0, 0, 2])
    assert np.all(res.get_column("query_indices") == [0, 1, 2])

    res = query.find_overlaps(subject, query_type="start")
    assert np.all(res.get_column("self_indices") == [])
    assert np.all(res.get_column("query_indices") == [])

    res = query.find_overlaps(subject, query_type="start", max_gap=1)
    assert np.all(res.get_column("self_indices") == [0, 0, 2])
    assert np.all(res.get_column("query_indices") == [0, 1, 2])

    res = query.find_overlaps(subject, query_type="end", select="first")
    assert np.all(res.get_column("self_indices") == [])
    assert np.all(res.get_column("query_indices") == [])

    res = query.find_overlaps(subject, query_type="within", max_gap=1)
    assert np.all(res.get_column("self_indices") == [])
    assert np.all(res.get_column("query_indices") == [])

def test_count_overlaps():
    query = IRanges([1, 4, 9], [5, 4, 2])
    subject = IRanges([2, 2, 10], [1, 2, 3])

    res = query.count_overlaps(subject)
    assert np.all(res == [1, 1, 1])

    res = query.count_overlaps(subject, max_gap=0)
    assert np.all(res == [1, 2, 1])

def test_subset_overlaps():
    subject = IRanges([1, 4, 9], [5, 4, 2])
    query = IRanges([2, 2, 10], [1, 2, 3])

    res = subject.subset_by_overlaps(query)
    assert len(res) == 2

    res = subject.subset_by_overlaps(query, query_type="within")
    assert len(res) == 1
    assert all(np.equal(res.start, [1]))
    assert all(np.equal(res.width, [5]))

    query = IRanges([1, 4], [5, 4])
    subject = IRanges([2, 2], [1, 2])

    res = query.subset_by_overlaps(subject)
    assert all(np.equal(res.start, [1]))
    assert all(np.equal(res.width, [5]))


def test_nearest():
    query = IRanges([1, 3, 9], [2, 5, 2])
    subject = IRanges([3, 5, 12], [1, 2, 1])

    res = subject.nearest(query, select="all")
    assert res == [[0], [0, 1], [2]]

    res = query.nearest(subject)
    assert res == [[0, 1], [1], [2]]


def test_precede():
    query = IRanges([1, 3, 9], [3, 5, 2])
    subject = IRanges([3, 2, 10], [1, 12, 3])

    res = subject.precede(query)
    assert res == [[2], [2], []]

    res = query.precede(subject)
    assert res == [[2], [], []]


def test_follow():
    query = IRanges([1, 3, 9], [3, 5, 2])
    subject = IRanges([3, 2, 10], [1, 12, 3])

    res = subject.follow(query)
    assert res == [[], [], [0]]

    res = query.follow(subject)
    assert res == [[], [], [1]]


def test_distance():
    x = IRanges([1], [5])
    y = IRanges([7], [4])

    res = x.distance(y)
    assert res == [1]

    x = IRanges([1], [5])
    y = IRanges([6], [4])

    res = x.distance(y)
    assert res == [0]
