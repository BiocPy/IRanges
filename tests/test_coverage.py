import numpy as np
import pytest
from iranges import IRanges

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_coverage():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    cov = x.coverage()
    assert all(np.equal(cov, [3, 3, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1]))


def test_coverage_with_shift():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    cov = x.coverage(shift=7)
    assert all(
        np.equal(cov, [1, 1, 1, 0, 1, 1, 2, 3, 3, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1])
    )


def test_coverage_with_shift_and_width():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    cov = x.coverage(shift=7, width=27)
    assert all(
        np.equal(
            cov,
            [
                1,
                1,
                1,
                0,
                1,
                1,
                2,
                3,
                3,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                2,
                2,
                2,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )
    )


def test_coverage_with_weight():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    cov = x.coverage(weight=10)
    assert all(
        np.equal(cov, [x * 10 for x in [3, 3, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1]])
    )
