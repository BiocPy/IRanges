# Changelog

## Version 0.2.10

- Added a numpy vectorized version of finding gaps (tldr: not fast compared to the traditional version). May be needs a better implementation
- Added NCLS based intersection operation (based on what pyranges does in their internals)
- Added tests for intersection operations.

## Version 0.2.8 - 0.2.9

Optimizing a couple of methods in `IRanges`:

- Update `gaps` and `reduce` to slightly faster NumPy based operations.
- Switch `np.array` to `np.asarray`

## Version 0.2.7

Changes to be compatible with NumPy's 2.0 release:

> If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).
> For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.

## Version 0.2.4

- Support coercion from `IRanges` to Polars and vice-versa
- Setting up `myst_nb` to execute snippets in tutorial/documentation markdown files

## Version 0.2.3

- Feature complete implementation of IRanges from Bioconductor
