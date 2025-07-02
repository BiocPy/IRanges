# Changelog

## Version 0.5.0 - 0.5.2

- Switch to [LTLA/nclist-cpp](https://github.com/LTLA/nclist-cpp) for overlap and intersection operations.
- Update Github template and setup.py to support editable builds.
- Improving performance of search operations - nearest, follow and precede.

## Version 0.4.2

- Fixes edge cases for overlap and search methods.
- Add tile and sliding window operations.
- Updates tests to capture most scenarios.
- Update docstrings and edit tutorials.

## Version 0.4.0 - 0.4.1

This is a major rewrite of the package partly driven by performance and for better consistency with the IRanges R/Bioconductor package.

- Using pybind11, reimplement inter-range methods to a cpp implementation using code derived from the R package.
- Better Start/End/Width resolver that is similar to the R implementation.
- Overhaul of many of the find overlaps and search methods to better align with R's expectations of how these methods work.
- Update docstrings, more tests.

## Version 0.3.0

- chore: Remove Python 3.8 (EOL)
- precommit: Replace docformatter with ruff's formatter

## Version 0.2.10 - 0.2.12

- Added a numpy vectorized version of finding gaps (tldr: not fast compared to the traditional version). May be needs a better implementation
- Added NCLS based intersection operation (based on what pyranges does in their internals)
- Added tests for intersection operations.
- Fixed and issue when coercing `IRanges` containing mcols.

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
