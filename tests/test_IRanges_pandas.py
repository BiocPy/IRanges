import numpy as np
import pandas as pd
from iranges import IRanges
from biocframe import BiocFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_pandas():
    df = pd.DataFrame({"start": [1, 2, 3, 4], "width": [4, 5, 6, 7]})

    x = IRanges.from_pandas(df)
    assert (x.get_start() == np.array(df["start"])).all()
    assert (x.get_width() == np.array(df["width"])).all()
    assert isinstance(x.mcols, BiocFrame)


def test_pandas_export():
    x = IRanges([1, 2, 3, 4], [4, 5, 6, 7])

    y = x.to_pandas()
    assert y is not None
    assert isinstance(y, pd.DataFrame)
    assert set(y.columns.tolist()).issubset(["starts", "ends", "widths"])
