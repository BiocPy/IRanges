import numpy as np
import pandas as pd
import pytest
from iranges import IRanges, Irange

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_init_from_lists():
    ranges1 = IRanges([1, 10, 20], [5, 8, 5])

    assert ranges1 is not None

def test_init_numpy():
    starts = np.array([5, 15, 25])
    widths = np.array([3, 6, 4])
    ranges2 = IRanges(starts, widths)

    assert ranges2 is not None

def test_init_pandas():
    df = pd.DataFrame({
        'start': [2, 12, 22],
        'width': [4, 7, 3]
    })
    ranges3 = IRanges.from_dataframe(df)

    assert ranges3 is not None

# # Operations
# # Find overlaps
# overlaps = ranges1.find_overlaps(ranges2)

# # Count overlaps
# counts = ranges1.count_overlaps(ranges2)

# # Merge overlapping ranges
# merged = ranges1.merge(min_gap=2)

# # Convert to DataFrame for analysis
# df