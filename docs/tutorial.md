---
file_format: mystnb
kernelspec:
  name: python
---


# Integer ranges in Python

Python implementation of the [**IRanges**](https://bioconductor.org/packages/IRanges) Bioconductor package.

To get started, install the package from [PyPI](https://pypi.org/project/IRanges/)

```bash
pip install iranges

# To install optional dependencies
pip install iranges[optional]
```

# Constuct `IRanges`

An `IRanges` holds a **start** position and a **width**, and is most typically used to represent coordinates along some genomic sequence. The interpretation of the start position depends on the application; for sequences, the start is usually a 1-based position, but other use cases may allow zero or even negative values.

```{code-cell}
from iranges import IRanges

starts = [1, 2, 3, 4]
widths = [4, 5, 6, 7]
x = IRanges(starts, widths)

print(x)
```

# Interval Operations

`IRanges` supports most interval based operations. For example to compute gaps

```{code-cell}

x = IRanges([-2, 6, 9, -4, 1, 0, -6, 10], [5, 0, 6, 1, 4, 3, 2, 3])

gaps = x.gaps()
print(gaps)
```

Or Perform interval set operations

```{code-cell}
x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])
y = IRanges([14, 0, -5, 6, 18], [7, 3, 8, 3, 3])

intersection = x.intersect(y)
print(intersection)
```


# Overlap operations

IRanges uses [nested containment lists](https://github.com/pyranges/ncls) under the hood to perform fast overlap and search based operations. These methods typically return a list of indices that map to each interval in query.

```{code-cell}
subject = IRanges([2, 2, 10], [1, 2, 3])
query = IRanges([1, 4, 9], [5, 4, 2])

overlap = subject.find_overlaps(query)
print(overlap)
```

Similarly one can perform search operations like follow, precede or nearest.

```{code-cell}
query = IRanges([1, 3, 9], [2, 5, 2])
subject = IRanges([3, 5, 12], [1, 2, 1])

nearest = subject.nearest(query, select="all")
print(nearest)
```

# More Information

- [IRanges reference](https://biocpy.github.io/IRanges/api/iranges.html#iranges-package)
- [Bioc/IRanges](https://bioconductor.org/packages/release/bioc/html/IRanges.html)


<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
