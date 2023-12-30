import copy

import numpy as np
import pytest
from biocframe import BiocFrame
from biocutils import combine_sequences
from iranges import IRanges

__author__ = "Aaron Lun"
__copyright__ = "LTLA"
__license__ = "MIT"


def test_IRanges_basic():
    starts = [1, 2, 3, 4]
    widths = [4, 5, 6, 7]
    x = IRanges(starts, widths)
    assert (x.get_start() == np.array(starts)).all()
    assert (x.get_width() == np.array(widths)).all()

    y = x.set_start([0, 1, 2, 3])
    assert (y.get_start() == np.array([0, 1, 2, 3])).all()
    y = x.set_width([10, 11, 12, 13])
    assert (y.get_width() == np.array([10, 11, 12, 13])).all()

    # Works with NumPy array inputs.
    x = IRanges(np.array(starts, dtype=np.int32), np.array(widths, dtype=np.int32))
    assert (x.get_start() == np.array(starts)).all()
    assert (x.get_width() == np.array(widths)).all()

    # Throws an error appropriately.
    with pytest.raises(ValueError) as ex:
        IRanges([], [1])
    assert str(ex.value).find("should have the same length") >= 0

    with pytest.raises(ValueError) as ex:
        x.set_start([1])
    assert str(ex.value).find("should be equal to 'length(<IRanges>)'") >= 0

    with pytest.raises(ValueError) as ex:
        IRanges([1], [-1])
    assert str(ex.value).find("must be non-negative") >= 0

    with pytest.raises(ValueError) as ex:
        IRanges([1], [2**31 - 1])
    assert str(ex.value).find("should fit in a 32-bit") >= 0

    # Adding names.
    x = IRanges(starts, widths, names=["A", "B", "C", "D"])
    assert x.get_names() == ["A", "B", "C", "D"]
    y = x.set_names(None)
    assert y.get_names() is None
    y = x.set_names(["a", "b", "c", "d"])
    assert y.get_names() == ["a", "b", "c", "d"]


def test_IRanges_metadata():
    starts = [1, 2, 3, 4]
    widths = [4, 5, 6, 7]
    x = IRanges(starts, widths)
    assert x.get_mcols().shape[1] == 0

    y = x.set_mcols(BiocFrame({"ok": [True, False, False, True]}))
    assert y.get_mcols().column("ok") == [True, False, False, True]
    y = x.set_mcols(None)
    assert y.get_mcols().shape[1] == 0

    with pytest.raises(TypeError) as ex:
        IRanges(starts, widths, mcols={})
    assert str(ex.value).find("should be a BiocFrame") >= 0

    with pytest.raises(ValueError) as ex:
        IRanges(starts, widths, mcols=BiocFrame({}, number_of_rows=3))
    assert str(ex.value).find("Number of rows") >= 0

    assert x.get_metadata() == {}
    y = x.set_metadata({"A": 2})
    assert "A" in y.get_metadata()
    y = x.set_metadata([])
    assert y.get_metadata() == {}
    y = x.set_metadata(None)
    assert y.get_metadata() == {}


def test_IRanges_getitem():
    starts = [1, 2, 3, 4]
    widths = [4, 5, 6, 7]
    x = IRanges(starts, widths)

    y = x[1:3]
    assert len(y) == 2
    assert (y.get_start() == np.array([2, 3])).all()
    assert (y.get_width() == np.array([5, 6])).all()

    y = x.set_names(["A", "B", "C", "D"])[[0, 3]]
    assert (y.get_start() == np.array([1, 4])).all()
    assert y.get_names() == ["A", "D"]

    y = x.set_mcols(BiocFrame({"ok": [True, True, False, False]}))[::-1]
    assert (y.get_start() == np.array([4, 3, 2, 1])).all()
    assert y.get_mcols().column("ok") == [False, False, True, True]

    y = x.set_metadata({"A": "B"})[0]
    assert (y.get_start() == np.array([1])).all()
    assert y.get_metadata() == {"A": "B"}


def test_IRanges_setitem():
    starts = [1, 2, 3, 4]
    widths = [4, 5, 6, 7]
    starts2 = [10, 20, 30, 40]
    widths2 = [50, 60, 70, 80]

    x = IRanges(starts, widths)
    y = IRanges(starts2, widths2)
    x[1:3] = y[1:3]
    assert (x.get_start() == np.array([1, 20, 30, 4])).all()
    assert (x.get_width() == np.array([4, 60, 70, 7])).all()
    assert x.get_names() is None

    x = IRanges(starts, widths, mcols=BiocFrame({"foo": ["a", "b", "c", "d"]}))
    y = IRanges(starts2, widths2, mcols=BiocFrame({"foo": ["A", "B", "C", "D"]}))
    x[1:3] = y[1:3]
    assert x.get_mcols().column("foo") == ["a", "B", "C", "d"]

    x = IRanges(starts, widths, names=["a", "b", "c", "d"])
    y = IRanges(starts2, widths2, names=["A", "B", "C", "D"])
    x[1:3] = y[1:3]
    assert x.get_names() == ["a", "B", "C", "d"]

    x = IRanges(starts, widths)
    y = IRanges(starts2, widths2, names=["A", "B", "C", "D"])
    x[1:3] = y[1:3]
    assert x.get_names() == ["", "B", "C", ""]

    x = IRanges(starts, widths, names=["a", "b", "c", "d"])
    y = IRanges(starts2, widths2)
    x[1:3] = y[1:3]
    assert x.get_names() == ["a", "", "", "d"]


def test_IRanges_print():
    starts = [1, 2, 3, 4]
    widths = [4, 5, 6, 7]
    x = IRanges(starts, widths)
    assert repr(x).startswith("IRanges(") >= 0
    assert str(x).startswith("IRanges ") >= 0


def test_IRanges_copy():
    starts = [1, 2, 3, 4]
    widths = [4, 5, 6, 7]
    x = IRanges(starts, widths)

    shallow = copy.copy(x)
    shallow.set_start([4, 3, 2, 1], in_place=True)
    assert x.get_start()[0] == 1
    assert shallow.get_start()[0] == 4

    deep = copy.deepcopy(x)
    deep.get_start()[0] = 2
    assert x.get_start()[0] == 1
    assert deep.get_start()[0] == 2


def test_IRanges_combine():
    starts = [1, 2, 3, 4]
    widths = [4, 5, 6, 7]
    starts2 = [10, 20, 30, 40]
    widths2 = [50, 60, 70, 80]

    x = IRanges(starts, widths)
    y = IRanges(starts2, widths2)
    comb = combine_sequences(x, y)
    assert (comb.get_start() == np.array([1, 2, 3, 4, 10, 20, 30, 40])).all()
    assert (comb.get_width() == np.array([4, 5, 6, 7, 50, 60, 70, 80])).all()
    assert comb.get_names() is None

    x = IRanges(starts, widths, mcols=BiocFrame({"foo": ["a", "b", "c", "d"]}))
    y = IRanges(starts2, widths2, mcols=BiocFrame({"foo": ["A", "B", "C", "D"]}))
    comb = combine_sequences(x, y)
    assert comb.get_mcols().column("foo") == ["a", "b", "c", "d", "A", "B", "C", "D"]

    x = IRanges(starts, widths, names=["a", "b", "c", "d"])
    y = IRanges(starts2, widths2, names=["A", "B", "C", "D"])
    comb = combine_sequences(x, y)
    assert comb.get_names() == ["a", "b", "c", "d", "A", "B", "C", "D"]

    x = IRanges(starts, widths)
    y = IRanges(starts2, widths2, names=["A", "B", "C", "D"])
    comb = combine_sequences(x, y)
    assert comb.get_names() == ["", "", "", "", "A", "B", "C", "D"]


def test_empty():
    r = IRanges.empty()

    assert r is not None
    assert isinstance(r, IRanges)

    subset = r[1:10]
    assert subset is not None
    assert isinstance(subset, IRanges)
