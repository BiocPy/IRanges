import numpy as np
import pytest
from compressed_lists import splitAsCompressedList

from iranges import CompressedIRangesList, IRanges

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def irange_data():
    range1 = IRanges(start=[1, 2, 3], width=[5, 2, 8])
    range2 = IRanges(start=[15, 45, 20, 1], width=[15, 100, 80, 5])
    return range1, range2


def test_creation(irange_data):
    range_list = CompressedIRangesList.from_list([irange_data[0], irange_data[1]], names=["a", "b"])

    assert isinstance(range_list, CompressedIRangesList)
    assert len(range_list) == 2
    assert isinstance(range_list.unlist_data, IRanges)
    assert len(range_list.get_unlist_data()) == 7
    assert list(range_list.get_element_lengths()) == [3, 4]
    assert np.allclose(range_list[0].get_start(), [1, 2, 3])


def test_split_iranges(irange_data):
    range_list = CompressedIRangesList.from_list([irange_data[0], irange_data[1]], names=["a", "b"])

    clist = splitAsCompressedList(range_list.unlist_data, groups_or_partitions=[0, 1, 2, 0, 0, 1, 1])

    assert isinstance(clist, CompressedIRangesList)
    assert len(clist) == 3

    val = clist.__repr__()
    assert isinstance(val, str)

    val = clist.__str__()
    assert isinstance(val, str)
