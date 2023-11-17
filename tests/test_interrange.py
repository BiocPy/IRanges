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


def test_reduce():
    starts = [-2, 6, 9, -4, 1, 0, -6, 3, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 0, 3]
    x = IRanges(starts, widths)

    reduced = x.reduce(with_reverse_map=True)

    assert all(np.equal(reduced.start, [-6, -2, 6, 9]))
    assert all(np.equal(reduced.width, [3, 7, 0, 6]))
    assert reduced.mcols.colnames.as_list() == ["revmap"]


def test_reduce_drop_ranges():
    starts = [-2, 6, 9, -4, 1, 0, -6, 3, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 0, 3]
    x = IRanges(starts, widths)

    reduced = x.reduce(drop_empty_ranges=True)
    assert all(np.equal(reduced.start, [-6, -2, 9]))
    assert all(np.equal(reduced.width, [3, 7, 6]))
    assert reduced.mcols.colnames.as_list() == []


def test_reduce_drop_ranges_and_revmap():
    starts = [-2, 6, 9, -4, 1, 0, -6, 3, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 0, 3]
    x = IRanges(starts, widths)

    reduced = x.reduce(drop_empty_ranges=True, with_reverse_map=True)
    assert all(np.equal(reduced.start, [-6, -2, 9]))
    assert all(np.equal(reduced.width, [3, 7, 6]))
    assert reduced.mcols.colnames.as_list() == ["revmap"]


def test_gap():
    x = IRanges([-2, 6, 9, -4, 1, 0, -6, 10], [5, 0, 6, 1, 4, 3, 2, 3])

    gaps = x.gaps()
    assert all(np.equal(gaps.start, [-3, 5]))
    assert all(np.equal(gaps.width, [1, 4]))


def test_gap_with_restrictions():
    x = IRanges([-2, 6, 9, -4, 1, 0, -6, 10], [5, 0, 6, 1, 4, 3, 2, 3])

    gaps = x.gaps(start=-6, end=20)
    assert all(np.equal(gaps.start, [-3, 5, 15]))
    assert all(np.equal(gaps.width, [1, 4, 6]))


def test_disjoin():
    x = IRanges([-2, 6, 9, -4, 1, 0, -6, 10], [5, 0, 6, 1, 4, 3, 2, 3])

    dj = x.disjoin()
    assert all(np.equal(dj.start, [-6, -4, -2, 0, 1, 3, 9, 10, 13]))
    assert all(np.equal(dj.width, [2, 1, 2, 1, 2, 2, 1, 3, 2]))


def test_disjoin_with_revmap():
    x = IRanges([-2, 6, 9, -4, 1, 0, -6, 10], [5, 0, 6, 1, 4, 3, 2, 3])

    dj = x.disjoin(with_reverse_map=True)
    assert all(np.equal(dj.start, [-6, -4, -2, 0, 1, 3, 9, 10, 13]))
    assert all(np.equal(dj.width, [2, 1, 2, 1, 2, 2, 1, 3, 2]))
    assert dj.mcols.colnames.as_list() == ["revmap"]
