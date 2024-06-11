---
file_format: mystnb
kernelspec:
  name: python
---


# `IRanges`: Interval arithmetic

Python implementation of the [**IRanges**](https://bioconductor.org/packages/IRanges) Bioconductor package.

An `IRanges` holds a **start** position and a **width**, and is typically used to represent coordinates along a genomic sequence. The interpretation of the **start** position depends on the application; for sequences, the **start** is usually a 1-based position, but other use cases may allow zero or even negative values, e.g., circular genomes.

`IRanges` uses [nested containment lists](https://github.com/pyranges/ncls) under the hood to perform fast overlap and search based operations.

These classes follow a functional paradigm for accessing or setting properties, with further details discussed in [functional paradigm](https://biocpy.github.io/tutorial/chapters/philosophy.html) section.

## Installation
To get started, install the package from [PyPI](https://pypi.org/project/IRanges/)

```bash
pip install iranges
```

The descriptions for some of these methods come from the [Bioconductor documentation](https://bioconductor.org/packages/release/bioc/html/IRanges.html).

# Construct `IRanges`

An `IRanges` holds a **start** position and a **width**, and is most typically used to represent coordinates along some genomic sequence. The interpretation of the start position depends on the application; for sequences, the start is usually a 1-based position, but other use cases may allow zero or even negative values (e.g. circular genomes).

```{code-cell}
from iranges import IRanges

starts = [-2, 6, 9, -4, 1, 0, -6, 10]
widths = [5, 0, 6, 1, 4, 3, 2, 3]
ir = IRanges(starts, widths)

print(ir)
```

# Accessing properties

Properties can be accessed directly from the object:

```{code-cell}
print("Number of intervals:", len(ir))

print("start positions:", ir.get_start())
print("width of each interval:", ir.get_width())
print("end positions:", ir.get_end())
```

Just like BiocFrame, these classes offer both functional-style and property-based getters and setters.

```{code-cell}
print("start positions:", ir.start)
print("width of each interval:", ir.width)
print("end positions:", ir.end)
```

# Reduced ranges (Normality)

`reduce` method reduces the intervals to an `IRanges` where the intervals are:

- not empty
- not overlapping
- ordered from left to right
- not even adjacent (i.e. there must be a non empty gap between 2 consecutive ranges).

```{code-cell}
reduced = ir.reduce()
print(reduced)
```

# Overlap operations

`IRanges` uses [nested containment lists](https://github.com/pyranges/ncls) under the hood to perform fast overlap and search-based operations.

```{code-cell}
subject = IRanges([2, 2, 10], [1, 2, 3])
query = IRanges([1, 4, 9], [5, 4, 2])

overlap = subject.find_overlaps(query)
print(overlap)
```

## Finding neighboring ranges

The `nearest`, `precede` or `follow` methods finds the nearest overlapping range along the specified direction.

```{code-cell}
query = IRanges([1, 3, 9], [2, 5, 2])
subject = IRanges([3, 5, 12], [1, 2, 1])

nearest = subject.nearest(query, select="all")
print(nearest)
```

These methods typically return a list of indices from `subject` for each interval in `query`.

## coverage

The `coverage` method counts the number of overlaps for each position.

```{code-cell}
cov = subject.coverage()
print(cov)
```


# Transforming ranges

`shift` adjusts the start positions by their **shift**.

```{code-cell}
shifted = ir.shift(shift=10)
print(shifted)
```

Other range transformation methods include `narrow`, `resize`, `flank`, `reflect` and `restrict`. For example `narrow` supports the adjustment of `start`, `end` and `width` values, which should be relative to each range.

```{code-cell}
narrowed = ir.narrow(start=4, width=2)
print(narrowed)
```

## Disjoin intervals

Well as the name says, computes disjoint intervals.

```{code-cell}
disjoint = ir.disjoin()
print(disjoint)
```

## `reflect` and `flank`

`reflect` reverses each range within a set of common reference bounds.

```{code-cell}
starts = [2, 5, 1]
widths = [2, 3, 3]
x = IRanges(starts, widths)
bounds = IRanges([0, 5, 3], [11, 2, 7])

res = x.reflect(bounds=bounds)
print(res)
```

`flank` returns ranges of a specified width that flank, to the left (default) or right, each input range. One use case of this is forming promoter regions for a set of genes.

```{code-cell}
starts = [2, 5, 1]
widths = [2, 3, 3]
x = IRanges(starts, widths)

res = x.flank(2, start=False)
print(res)
```

# Set operations

`IRanges` supports most interval set operations. For example, to compute `gaps`:

```{code-cell}
gaps = ir.gaps()
print(gaps)
```

Or Perform interval set operations, e..g `union`, `intersection`, `disjoin`:

```{code-cell}
x = IRanges([1, 5, -2, 0, 14], [10, 5, 6, 12, 4])
y = IRanges([14, 0, -5, 6, 18], [7, 3, 8, 3, 3])

intersection = x.intersect(y)
print(intersection)
```

----

## Further reading

- [IRanges reference](https://biocpy.github.io/IRanges/api/iranges.html#iranges-package)
- [Bioc/IRanges](https://bioconductor.org/packages/release/bioc/html/IRanges.html)
