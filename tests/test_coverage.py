import numpy as np
import pytest
from iranges import IRanges


def test_coverage():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    print(x)

    cov = x.coverage()
    assert all(np.equal(cov, [3, 3, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1]))


def test_coverage_with_shift():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    print(x)

    cov = x.coverage(shift=7)

    print(x)
    assert all(
        np.equal(cov, [1, 1, 1, 0, 1, 1, 2, 3, 3, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1])
    )


def test_coverage_with_shift_and_width():
    starts = [-2, 6, 9, -4, 1, 0, -6, 10]
    widths = [5, 0, 6, 1, 4, 3, 2, 3]
    x = IRanges(starts, widths)

    print(x)

    cov = x.coverage(shift=7, width=27)

    print(x)
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

    print(x)
    cov = x.coverage(weight=10)

    print(x)
    assert all(
        np.equal(cov, [x * 10 for x in [3, 3, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1]])
    )
