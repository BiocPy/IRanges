import numpy as np
import polars as pl
from iranges import IRanges
from biocframe import BiocFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_from_polars():
    df = pl.DataFrame({"start": [1, 2, 3, 4], "width": [4, 5, 6, 7]})

    x = IRanges.from_polars(df)
    assert (x.get_start() == np.array(df["start"])).all()
    assert (x.get_width() == np.array(df["width"])).all()
    assert isinstance(x.mcols, BiocFrame)


def test_to_polars_export():
    x = IRanges([1, 2, 3, 4], [4, 5, 6, 7])

    y = x.to_polars()
    assert y is not None
    assert isinstance(y, pl.DataFrame)
    assert set(y.columns).issubset(["starts", "ends", "widths"])


def test_to_polars_names():
    x = IRanges(
        [1, 2, 3, 4], [4, 5, 6, 7], names=["range1", "range2", "range3", "range4"]
    )

    y = x.to_polars()
    assert y is not None
    assert isinstance(y, pl.DataFrame)
    assert set(y.columns).issubset(["starts", "ends", "widths", "names"])
