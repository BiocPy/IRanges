<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/IRanges.svg?branch=main)](https://cirrus-ci.com/github/<USER>/IRanges)
[![ReadTheDocs](https://readthedocs.org/projects/IRanges/badge/?version=latest)](https://IRanges.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/IRanges/main.svg)](https://coveralls.io/r/<USER>/IRanges)
[![PyPI-Server](https://img.shields.io/pypi/v/IRanges.svg)](https://pypi.org/project/IRanges/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/IRanges.svg)](https://anaconda.org/conda-forge/IRanges)
[![Monthly Downloads](https://pepy.tech/badge/IRanges/month)](https://pepy.tech/project/IRanges)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/IRanges)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/IRanges.svg)](https://pypi.org/project/IRanges/)
![Unit tests](https://github.com/BiocPy/IRanges/actions/workflows/pypi-test.yml/badge.svg)

# Integer ranges in Python

Python implementation of the [**IRanges**](https://bioconductor.org/packages/IRanges) Bioconductor package.

To get started, install the package from [PyPI](https://pypi.org/project/IRanges/)

```bash
pip install iranges

# To install optional dependencies
pip install iranges[optional]
```

## IRanges

An `IRanges` holds a **start** position and a **width**, and is most typically used to represent coordinates along some genomic sequence. The interpretation of the start position depends on the application; for sequences, the start is usually a 1-based position, but other use cases may allow zero or even negative values.

```python
from iranges import IRanges

starts = [1, 2, 3, 4]
widths = [4, 5, 6, 7]
x = IRanges(starts, widths)

print(x)
```

     ## output
     IRanges object with 4 ranges and 0 metadata columns
                    start              end            width
     <ndarray[int32]> <ndarray[int32]> <ndarray[int32]>
     [0]                1                5                4
     [1]                2                7                5
     [2]                3                9                6
     [3]                4               11                7


## Interval Operations

`IRanges` supports most interval based operations. For example to compute gaps

```python

x = IRanges([-2, 6, 9, -4, 1, 0, -6, 10], [5, 0, 6, 1, 4, 3, 2, 3])

gaps = x.gaps()
print(gaps)
```

     ## output

     IRanges object with 2 ranges and 0 metadata columns
                    start              end            width
     <ndarray[int32]> <ndarray[int32]> <ndarray[int32]>
     [0]               -3               -2                1
     [1]                5                9                4

Or Perform interval set operations

```python
x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])
y = IRanges([14, 0, -5, 6, 18], [7, 3, 8, 3, 3])

intersection = x.intersect(y)
print(intersection)
```

     ## output
     IRanges object with 3 ranges and 0 metadata columns
                    start              end            width
     <ndarray[int32]> <ndarray[int32]> <ndarray[int32]>
     [0]               -2                3                5
     [1]                6                9                3
     [2]               14               18                4

### Overlap operations

IRanges uses [nested containment lists](https://github.com/pyranges/ncls) under the hood to perform fast overlap and search based operations. These methods typically return a list of indices that map to each interval in query.

```python
subject = IRanges([2, 2, 10], [1, 2, 3])
query = IRanges([1, 4, 9], [5, 4, 2])

overlap = subject.find_overlaps(query)
print(overlap)
```

     ## output
     [[1, 0], [], [2]]

Similarly one can perform search operations like follow, precede or nearest.

```python
query = IRanges([1, 3, 9], [2, 5, 2])
subject = IRanges([3, 5, 12], [1, 2, 1])

nearest = subject.nearest(query, select="all")
print(nearest)
```

     ## output
     [[0], [0, 1], [2]]

## Further Information

- [IRanges reference](https://biocpy.github.io/IRanges/api/iranges.html#iranges-package)
- [Bioc/IRanges](https://bioconductor.org/packages/release/bioc/html/IRanges.html)


<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
