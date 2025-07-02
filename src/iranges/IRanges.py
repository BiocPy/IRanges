from copy import deepcopy
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import biocutils as ut
import numpy as np
from biocframe import BiocFrame
from biocutils import Names, combine_rows, combine_sequences, show_as_cell

from . import lib_iranges as libir
from .sew_handler import SEWWrangler
from .utils import compute_up_down, normalize_array

__author__ = "Aaron Lun, Jayaram Kancherla"
__copyright__ = "LTLA, jkanche"
__license__ = "MIT"


class IRangesIter:
    """An iterator to :py:class:`~iranges.IRanges.IRanges`.

    Args:
        obj:
            Object to iterate.
    """

    def __init__(self, obj: "IRanges") -> None:
        """Initialize the iterator.

        Args:
            obj:
                Source object to iterate.
        """
        self._iranges = obj
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index < len(self._iranges):
            iter_row_index = self._iranges.names[self._current_index] if self._iranges.names is not None else None

            iter_slice = self._iranges.get_row(self._current_index)
            self._current_index += 1
            return (iter_row_index, iter_slice)

        raise StopIteration


class IRanges:
    """A collection of integer ranges, equivalent to the ``IRanges`` class from the
    `Bioconductor package <https://bioconductor.org/packages/IRanges>`_ of the same name.
    It enables efficient storage and manipulation of genomic intervals defined by start
    positions and widths.

    Each range consists of a start position and width. For genomic sequences, the start is
    typically 1-based, though other applications may use zero or negative values. The width
    represents the length of the interval. Ends are inclusive.
    """

    def __init__(
        self,
        start: Sequence[int] = [],
        width: Sequence[int] = [],
        names: Optional[Sequence[str]] = None,
        mcols: Optional[BiocFrame] = None,
        metadata: Optional[dict] = None,
        validate: bool = True,
    ):
        """
        Args:
            start:
                Sequence of integers containing the start position for each
                range. All values should fall within the range that can be
                represented by a 32-bit signed integer.

            width:
                Sequence of integers containing the width for each range. This
                should be of the same length as ``start``. All values should be
                non-negative and fall within the range that can be represented
                by a 32-bit signed integer. Similarly, ``start + width`` should
                not exceed the range of a 32-bit signed integer.

            names:
                Sequence of strings containing the name for each range. This
                should have length equal to ``start`` and should only contain
                strings. If no names are present, None may be supplied instead.

            mcols:
                A data frame containing additional metadata columns for each range.
                This should have number of rows equal to the length of ``start``.
                If None, defaults to a zero-column data frame.

            metadata:
                Additional metadata. If None, defaults to an empty dictionary.

            validate:
                Whether to validate the arguments, internal use only.
        """

        self._start = self._sanitize_start(start)
        self._width = self._sanitize_width(width)
        self._names = self._sanitize_names(names)
        self._mcols = self._sanitize_mcols(mcols)
        self._metadata = self._sanitize_metadata(metadata)

        if validate:
            self._validate_width()
            self._validate_names()
            self._validate_mcols()
            self._validate_metadata()

    def _sanitize_start(self, start):
        arr = np.array(start, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError("'start' must be a 1-dimensional array")
        return arr

    def _sanitize_width(self, width):
        arr = np.array(width, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError("'width' must be a 1-dimensional array")
        return arr

    def _validate_width(self):
        if len(self._start) != len(self._width):
            raise ValueError("'widths' must have the same lengths as 'start'")

        if np.any(self._width < 0):
            raise ValueError("'width' values must be non-negative")

        # make this optional ??
        # if not allow_empty and np.any(widths <= 0):
        #     raise Exception("widths must be positive")

        end = self._start + self._width
        if (end < self._start).any() or np.any(end > np.iinfo(np.int32).max) or np.any(end < np.iinfo(np.int32).min):
            raise ValueError("end position should fit in a 32-bit signed integer")

    def _sanitize_names(self, names):
        if names is None:
            return None
        elif not isinstance(names, list):
            names = Names(names)

        return names

    def _validate_names(self):
        if self._names is None:
            return None

        if not ut.is_list_of_type(self._names, str):
            raise ValueError("'names' should be a list of strings")

        if len(self._names) != len(self._start):
            raise ValueError("'names' must have the same length as 'start'")

    def _sanitize_mcols(self, mcols):
        if mcols is None:
            return BiocFrame({}, number_of_rows=len(self._start))
        else:
            return mcols

    def _validate_mcols(self):
        if not isinstance(self._mcols, BiocFrame):
            raise TypeError("'mcols' must be a BiocFrame")

        if self._mcols.shape[0] != len(self._start):
            raise ValueError("'mcols' must have the same number of rows as the length of 'start'")

    def _sanitize_metadata(self, metadata):
        if metadata is None:
            return {}
        elif not isinstance(metadata, dict):
            metadata = dict(metadata)

        return metadata

    def _validate_metadata(self):
        if not isinstance(self._metadata, dict):
            raise TypeError("'metadata' must be a dictionary")

    ########################
    #### Getter/setters ####
    ########################

    def get_start(self) -> np.ndarray:
        """Get start positions.

        Returns:
            NumPy array of 32-bit signed integers containing the start
            positions for all ranges.
        """
        return self._start

    def set_start(self, start: Sequence[int], in_place: bool = False) -> "IRanges":
        """Modify start positions (in-place operation).

        Args:
            start:
                Sequence of start positions, see the constructor for details.

            in_place:
                Whether to modify the object in place.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            modified start positions. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        output = self._define_output(in_place)
        if len(start) != len(output._start):
            raise ValueError("length of 'start' should be equal to 'length(<IRanges>)'")

        output._start = output._sanitize_start(start)
        return output

    @property
    def start(self) -> np.ndarray:
        """Get start positions.

        Returns:
            NumPy array of 32-bit signed integers containing the start
            positions for all ranges.
        """
        return self.get_start()

    @start.setter
    def start(self, start: Sequence[int]):
        """Modify start positions (in-place operation).

        Args:
            start:
                Sequence of start positions, see the constructor for details.
        """
        warn(
            "Setting property 'start'is an in-place operation, use 'set_start' instead",
            UserWarning,
        )

        self.set_start(start, in_place=True)

    def get_width(self) -> np.ndarray:
        """Get widths.

        Returns:
            NumPy array of 32-bit signed integers containing the widths for all
            ranges.
        """
        return self._width

    def set_width(self, width: Sequence[int], in_place: bool = False) -> "IRanges":
        """
        Args:
            width:
                Sequence of widths, see the constructor for details.

            in_place:
                Whether to modify the object in place.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            modified widths. Otherwise, the current object is directly modified
            and a reference to it is returned.
        """
        output = self._define_output(in_place)
        if len(width) != len(output._width):
            raise ValueError("length of 'width' should be equal to 'length(<IRanges>)'")

        output._width = output._sanitize_width(width)
        output._validate_width()
        return output

    @property
    def width(self) -> np.ndarray:
        """Get widths.

        Returns:
            NumPy array of 32-bit signed integers containing the widths for all
            ranges.
        """
        return self.get_width()

    @width.setter
    def width(self, width: Sequence[int]):
        """Set or modify width of each interval (in-place operation).

        Args:
            width:
                Sequence of widths, see the constructor for details.
        """
        warn(
            "Setting property 'width'is an in-place operation, use 'set_width' instead",
            UserWarning,
        )
        return self.set_width(width, in_place=True)

    def get_end(self) -> np.ndarray:
        """Get end positions (inclusive).

        Returns:
            NumPy array of 32-bit signed integers containing the end position
            for all ranges.
        """
        return self._start + self._width - 1

    @property
    def end(self) -> np.ndarray:
        """Get all end positions (read-only).

        Returns:
            NumPy array of 32-bit signed integers containing the end position
            for all ranges.
        """
        return self.get_end()

    def get_names(self) -> Optional[Names]:
        """Get range names.

        Returns:
            List containing the names for all ranges, or None if no names are
            present.
        """
        return self._names

    def set_names(self, names: Optional[Sequence[str]], in_place: bool = False) -> "IRanges":
        """
        Args:
            names:
                Sequence of names or None, see the constructor for details.

            in_place:
                Whether to modify the object in place.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            modified names. Otherwise, the current object is directly modified
            and a reference to it is returned.
        """
        output = self._define_output(in_place)
        output._names = output._sanitize_names(names)
        output._validate_names()
        return output

    @property
    def names(self) -> Optional[Names]:
        """Get names.

        Returns:
            List containing the names for all ranges, or None if no names are
            available.
        """
        return self.get_names()

    @names.setter
    def names(self, names: Optional[Sequence[str]]):
        """Set new names (in-place operation).

        Args:
            names:
                Sequence of names or None, see the constructor for details.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            modified names. Otherwise, the current object is directly modified
            and a reference to it is returned.
        """
        warn(
            "Setting property 'names'is an in-place operation, use 'set_names' instead",
            UserWarning,
        )
        self.set_names(names, in_place=True)

    def get_mcols(self) -> BiocFrame:
        """Get metadata about ranges.

        Returns:
            Data frame containing additional metadata columns for all ranges.
        """
        return self._mcols

    def set_mcols(self, mcols: Optional[BiocFrame], in_place: bool = False) -> "IRanges":
        """Set new metadata about ranges.

        Args:
            mcols:
                Data frame of additional columns, see the constructor for
                details.

            in_place:
                Whether to modify the object in place.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            modified metadata columns. Otherwise, the current object is
            directly modified and a reference to it is returned.
        """
        output = self._define_output(in_place)
        output._mcols = output._sanitize_mcols(mcols)
        output._validate_mcols()
        return output

    @property
    def mcols(self) -> BiocFrame:
        """Get metadata.

        Returns:
            Data frame containing additional metadata columns for all ranges.
        """
        return self.get_mcols()

    @mcols.setter
    def mcols(self, mcols: Optional[BiocFrame]):
        """Set new metadata about ranges (in-place operation).

        Args:
            mcols:
                Data frame of additional columns, see the constructor for
                details.
        """
        warn(
            "Setting property 'mcols'is an in-place operation, use 'set_mcols' instead",
            UserWarning,
        )
        self.set_mcols(mcols, in_place=True)

    def get_metadata(self) -> dict:
        """Get additional metadata.

        Returns:
            Dictionary containing additional metadata.
        """
        return self._metadata

    def set_metadata(self, metadata: Optional[dict], in_place: bool = False) -> "IRanges":
        """Set or replace metadata.

        Args:
            metadata:
                Additional metadata.

            in_place:
                Whether to modify the object in place.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            modified metadata. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        output = self._define_output(in_place)
        output._metadata = output._sanitize_metadata(metadata)
        output._validate_metadata()
        return output

    @property
    def metadata(self) -> dict:
        """Get additional metadata.

        Returns:
            Dictionary containing additional metadata.
        """
        return self.get_metadata()

    @metadata.setter
    def metadata(self, metadata: Optional[dict]):
        """Set or replace metadata (in-place operation).

        Args:
            metadata:
                Additional metadata.
        """
        warn(
            "Setting property 'metadata'is an in-place operation, use 'set_metadata' instead",
            UserWarning,
        )
        self.set_metadata(metadata, in_place=True)

    #########################
    #### Getitem/setitem ####
    #########################

    def __len__(self) -> int:
        """
        Returns:
            Length of this object.
        """
        return len(self._start)

    def __getitem__(self, subset: Union[Sequence, int, str, bool, slice, range]) -> "IRanges":
        """Subset the IRanges.

        Args:
            subset:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

        Returns:
            A new ``IRanges`` object containing the ranges of interest.
        """
        idx, _ = ut.normalize_subscript(subset, len(self), self._names)
        return type(self)(
            start=ut.subset_sequence(self._start, idx),
            width=ut.subset_sequence(self._width, idx),
            names=ut.subset(self._names, idx) if self._names is not None else None,
            mcols=self._mcols[list(idx), :],
            metadata=self._metadata,
        )

    def __setitem__(self, args: Union[Sequence, int, str, bool, slice, range], value: "IRanges"):
        """Add or update positions (in-place operation).

        Args:
            subset:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be replaced, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

            value:
                An ``IRanges`` object of length equal to the number of ranges
                to be replaced, as specified by ``subset``.

        Returns:
            Specified ranges are replaced by ``value`` in the current object.
        """
        idx, _ = ut.normalize_subscript(args, len(self), self._names)
        self._start[idx] = value._start
        self._width[idx] = value._width
        self._mcols[idx, :] = value._mcols

        if value._names is not None:
            if self._names is None:
                self._names = [""] * len(self)
            for i, j in enumerate(idx):
                self._names[j] = value._names[i]
        elif self._names is not None:
            for i, j in enumerate(idx):
                self._names[j] = ""

    def get_row(self, index_or_name: Union[str, int]) -> "IRanges":
        """Access a row by index or row name.

        Args:
            index_or_name:
                Integer index of the row to access.

                Alternatively, you may provide a string specifying the row name to access,
                only if :py:attr:`~iranges.IRanges.IRanges.names` are available.

        Raises:
            ValueError:
                If ``index_or_name`` is not in row names.
                If the integer index is greater than the number of rows.

            TypeError:
                If ``index_or_name`` is neither a string nor an integer.

        Returns:
            IRanges: A sliced IRanges object.
        """

        if not isinstance(index_or_name, (int, str)):
            raise TypeError("`index_or_name` must be either an integer index or name.")

        return self[index_or_name]

    def __iter__(self) -> IRangesIter:
        """Iterator over ranges."""
        return IRangesIter(self)

    ##################
    #### Printing ####
    ##################

    def __repr__(self) -> str:
        with np.printoptions(threshold=50, edgeitems=3):
            message = "IRanges(start=" + repr(self._start)
            message += ", width=" + repr(self._width)
            if self._names:
                message += ", names=" + ut.print_truncated_list(self._names)

            if self._mcols.shape[1] > 0:
                message += ", mcols=" + repr(self._mcols)

            if len(self._metadata):
                message += ", metadata=" + ut.print_truncated_dict(self._metadata)

            message += ")"

        return message

    def __str__(self) -> str:
        nranges = len(self)
        nmcols = self._mcols.shape[1]
        output = (
            "IRanges object with "
            + str(nranges)
            + " range"
            + ("" if nranges == 1 else "s")
            + " and "
            + str(nmcols)
            + " metadata column"
            + ("" if nmcols == 1 else "s")
            + "\n"
        )

        added_table = False
        if nranges:
            if nranges <= 10:
                indices = range(nranges)
                insert_ellipsis = False
            else:
                indices = [0, 1, 2, nranges - 3, nranges - 2, nranges - 1]
                insert_ellipsis = True

            raw_floating = ut.create_floating_names(self._names, indices)
            if insert_ellipsis:
                raw_floating = raw_floating[:3] + [""] + raw_floating[3:]
            floating = ["", ""] + raw_floating

            columns = []

            sub_start = self._start[indices]
            sub_width = self._width[indices]
            sub_end = sub_start + sub_width - 1
            for prop, val in [
                ("start", sub_start),
                ("end", sub_end),
                ("width", sub_width),
            ]:
                header = [prop, "<" + ut.print_type(val) + ">"]
                showed = show_as_cell(val, range(len(val)))
                if insert_ellipsis:
                    showed = showed[:3] + ["..."] + showed[3:]
                columns.append(header + showed)

            if self._mcols.shape[1] > 0:
                spacer = ["|"] * (len(indices) + insert_ellipsis)
                columns.append(["", ""] + spacer)

                for col in self._mcols.get_column_names():
                    data = self._mcols.column(col)
                    showed = show_as_cell(data, indices)
                    header = [col, "<" + ut.print_type(data) + ">"]
                    showed = ut.truncate_strings(showed, width=max(40, len(header[0]), len(header[1])))
                    if insert_ellipsis:
                        showed = showed[:3] + ["..."] + showed[3:]
                    columns.append(header + showed)

            output += ut.print_wrapped_table(columns, floating_names=floating)
            added_table = True

        footer = []
        if len(self._metadata):
            footer.append(
                "metadata("
                + str(len(self._metadata))
                + "): "
                + ut.print_truncated_list(
                    list(self._metadata.keys()),
                    sep=" ",
                    include_brackets=False,
                    transform=lambda y: y,
                )
            )
        if len(footer):
            if added_table:
                output += "\n------\n"
            output += "\n".join(footer)

        return output

    #################
    #### Copying ####
    #################

    def _define_output(self, in_place):
        if in_place:
            return self
        else:
            return self.__copy__()

    def __copy__(self) -> "IRanges":
        """Shallow copy of the object.

        Returns:
            Same type as the caller, a shallow copy of this object.
        """
        return type(self)(
            start=self._start,
            width=self._width,
            names=self._names,
            mcols=self._mcols,
            metadata=self._metadata,
            validate=False,
        )

    def __deepcopy__(self, memo) -> "IRanges":
        """Deep copy of the object.

        Args:
            memo: Passed to internal :py:meth:`~deepcopy` calls.

        Returns:
            Same type as the caller, a deep copy of this object.
        """
        return type(self)(
            start=deepcopy(self._start, memo),
            width=deepcopy(self._width, memo),
            names=deepcopy(self._names, memo),
            mcols=deepcopy(self._mcols, memo),
            metadata=deepcopy(self._metadata, memo),
            validate=False,
        )

    #############################
    #### inter-range methods ####
    #############################

    def shift_and_clip_ranges(
        self, shift: np.ndarray, width: Union[int, None] = None, circle_length: Union[int, None] = None
    ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        """Shift and clip interval ranges.

        Args:
            shift:
                Array of shift values (will be recycled if necessary).

            width:
                Maximum width to clip to.
                Defaults to None for no clipping.

            circle_length:
                Length of circular sequence.
                Defaults to None for linear sequence.

        Returns:
            Tuple of:
            - Array of shifted/clipped start positions
            - Array of shifted/clipped widths
            - Coverage length
            - Boolean indicating if ranges are in tiling configuration
        """

        return libir.shift_and_clip_ranges(self._start, self._width, shift, width, circle_length)

    def coverage(
        self,
        shift: Optional[np.ndarray] = None,
        width: Union[int, None] = None,
        weight: Optional[np.ndarray] = None,
        circle_length: Union[int, None] = None,
        method: Literal["auto", "sort", "hash", "naive"] = "auto",
    ) -> np.ndarray:
        """Compute weighted coverage of ranges.

        Args:
            shift:
                Array of shift values.
                Defaults to None for no shift.

            width:
                Maximum width to clip to.
                Defaults to None for no clipping.

            weight:
                Array of weights.
                Defaults to None for equal weights
                for all ranges `(weight = 1)`.

            circle_length:
                Length of circular sequence.
                Defaults to None for linear sequence.

            method:
                Coverage computation method.
                Defaults to "auto".

        Returns:
            NumPy array containing coverage values.
        """
        if shift is None:
            shift = np.zeros(len(self))

        shift = normalize_array(shift, len(self))

        if weight is None:
            weight = np.ones(len(self), dtype=np.int32)

        weight = normalize_array(weight, len(self))

        return libir.coverage(self._start, self._width, shift, width, weight, circle_length, method)

    def range(self) -> "IRanges":
        """Concatenate and compute the mix and max across all ranges.

        Returns:
            An new ``IRanges`` instance with a single range,
            the minimum of all the start positions,  Maximum of all end positions.
        """

        min_start = self.start.min()
        max_end = self.end.max()

        return IRanges([min_start], [max_end - min_start + 1])

    def reduce(
        self,
        with_reverse_map: bool = False,
        drop_empty_ranges: bool = False,
        min_gap_width: int = 1,
    ) -> "IRanges":
        """Reduce orders the ranges, then merges overlapping or adjacent ranges.

        Args:
            with_reverse_map:
                Whether to return map of indices back to
                original object. Defaults to False.

            drop_empty_ranges:
                Whether to drop empty ranges. Defaults to False.

            min_gap_width:
                Ranges separated by a gap of at least ``min_gap_width``
                positions are not merged. Defaults to 1.

        Returns:
            A new ``IRanges`` object with reduced ranges.
        """
        if min_gap_width < 0:
            raise ValueError("'min_gap_width' cannot be negative.")

        reduced = libir.reduce_ranges(
            starts=self._start,
            widths=self._width,
            drop_empty_ranges=drop_empty_ranges,
            min_gapwidth=min_gap_width,
            with_revmap=with_reverse_map,
            with_inframe_start=False,
        )

        result = IRanges(reduced["start"], reduced["width"])
        if with_reverse_map:
            result._mcols.set_column("revmap", reduced["revmap"], in_place=True)

        return result

    def order(self, decreasing: bool = False) -> np.ndarray:
        """Get the order of indices for sorting.

        Args:
            decreasing:
                Whether to sort in descending order. Defaults to False.

        Returns:
            NumPy vector containing index positions in the sorted order.
        """
        order_buf = libir.get_order(self._start, self._width)

        if decreasing:
            return np.asarray(order_buf[::-1])

        return np.asarray(order_buf)

    def sort(self, decreasing: bool = False, in_place: bool = False) -> "IRanges":
        """Sort the ranges.

        Args:
            decreasing:
                Whether to sort in descending order.
                Defaults to False.

            in_place:
                Whether to modify the object in place.
                Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            sorted ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        order = self.order(decreasing=decreasing)
        output = self._define_output(in_place)
        return output[order]

    def gaps(self, start: Optional[int] = None, end: Optional[int] = None) -> "IRanges":
        """Gaps returns an ``IRanges`` object representing the set of intervals that remain after the ranges are
        removed specified by the start and end arguments.

        Args:
            start:
                Restrict start position. Defaults to 1.

            end:
                Restrict end position. Defaults to None.

        Returns:
            A new ``IRanges``'s with the gap regions.
        """
        gap_starts, gap_widths = libir.gaps_ranges(self._start, self._width, restrict_start=start, restrict_end=end)

        return IRanges(gap_starts, gap_widths)

    # follows the same logic as in https://stackoverflow.com/questions/55480499/split-set-of-intervals-into-minimal-set-of-disjoint-intervals
    # otherwise too much magic happening here - https://github.com/Bioconductor/IRanges/blob/devel/R/inter-range-methods.R#L389
    def disjoin(self, with_reverse_map: bool = False) -> "IRanges":
        """Calculate disjoint ranges.

        Args:
            with_reverse_map:
                Whether to return a map of indices back to the original object.
                Defaults to False.

        Returns:
           A new `IRanges` containing disjoint ranges.
        """

        ends = self.get_end()
        unique_starts = np.unique(self._start)
        unique_ends = np.unique(ends)

        adj_starts_temp = np.concatenate([unique_starts, unique_ends + 1])
        adj_starts = np.unique(adj_starts_temp)[:-1]

        adj_ends_temp = np.concatenate([unique_ends, unique_starts - 1])
        adj_ends = np.sort(np.unique(adj_ends_temp))[1:]

        adj_widths = adj_ends - adj_starts + 1

        # Find overlaps with original ranges
        adj_indices = []
        original_indices = []

        for i in range(len(adj_starts)):
            adj_start = adj_starts[i]
            adj_end = adj_ends[i]

            # Find overlapping original ranges
            overlaps = (self._start <= adj_end) & (ends >= adj_start)
            if np.any(overlaps):
                adj_indices.append(i)
                original_indices.extend(np.where(overlaps)[0])

        # Subset to only ranges that overlap with original
        adj_starts = adj_starts[adj_indices]
        adj_widths = adj_widths[adj_indices]

        result = IRanges(adj_starts, adj_widths)

        if with_reverse_map is True:
            mapping = []
            for i, adj_idx in enumerate(adj_indices):
                adj_start = adj_starts[i]
                adj_end = adj_starts[i] + adj_widths[i] - 1
                overlaps = np.where((self._start <= adj_end) & (ends >= adj_start))[0]
                mapping.append(overlaps)

            result._mcols.set_column("revmap", mapping, in_place=True)

        return result

    def is_disjoint(self) -> bool:
        """Check if the ranges are disjoint.

        Returns:
            True if all ranges are non-overlapping, otherwise False.
        """
        if len(self) < 2:
            return True

        ends = self.get_end()

        oo = self.order()
        sorted_start = self._start[oo]
        sorted_end = ends[oo]

        return bool(np.all(sorted_start[1:] > sorted_end[:-1]))

    def disjoint_bins(self) -> np.ndarray:
        """Split ranges into a set of bins so that the ranges in each bin are disjoint.

        Returns:
            An NumPy vector indicating the bin index for each range.
        """
        order = self.order()
        result = libir.disjoint_bins(self._start[order], self._width[order])

        return result[np.argsort(order, stable=True)]

    #############################
    #### intra-range methods ####
    #############################

    def shift(self, shift: Union[int, List[int], np.ndarray], in_place: bool = False) -> "IRanges":
        """Shift ranges by specified amount.

        Args:
            shift:
                Amount to shift by.

            in_place:
                Whether to modify the object in place. Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            shifted ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        output = self._define_output(in_place)

        shift_arr = normalize_array(shift, len(output))
        if shift_arr.mask.any():
            raise Exception("'shift' cannot contain NAs")

        new_starts = output._start.copy()
        if len(shift_arr) == 1:
            new_starts = output._start + shift_arr[0]
        else:
            new_starts = output._start + shift_arr.data

        output._start = new_starts
        return output

    def narrow(
        self,
        start: Optional[Union[int, List[int], np.ndarray]] = None,
        width: Optional[Union[int, List[int], np.ndarray]] = None,
        end: Optional[Union[int, List[int], np.ndarray]] = None,
        in_place: bool = False,
    ) -> "IRanges":
        """Narrow ranges.

        Important: These arguments are relative shift in positions for each range.

        Args:
            start:
                Relative start position.
                Defaults to None.

            width:
                Width of each interval position.
                Defaults to None.

            end:
                Relative end position.
                Defaults to None.

            in_place:
                Whether to modify the object in place.
                Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            narrowed ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        output = self._define_output(in_place)

        if len(output) == 0:
            return output

        sew = SEWWrangler(output._width, start, end, width, translate_negative=True, allow_nonnarrowing=False)
        window_starts, window_widths = sew.solve()

        output._start = output._start + window_starts - 1
        output._width = window_widths
        return output

    def resize(
        self,
        width: Union[int, List[int], np.ndarray],
        fix: Union[Literal["start", "end", "center"], List[Literal["start", "end", "center"]]] = "start",
        in_place: bool = False,
    ) -> "IRanges":
        """Resize ranges to the specified ``width`` where either the ``start``, ``end``, or ``center`` is used as an
        anchor.

        Args:
            width:
                Width to resize, must be non-negative!

            fix:
                Fix positions by "start", "end", or "center".

                Alternatively, `fix` may be a list with the same size
                as this `IRanges` object, denoting what to use as an
                anchor for each interval.

                Defaults to "start".

            in_place:
                Whether to modify the object in place.
                Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            resized ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        output = self._define_output(in_place)

        width_arr = normalize_array(width, len(output))
        if width_arr.mask.any() or np.any(width_arr < 0):
            raise Exception("'width' must be non-negative without NAs")

        if isinstance(fix, str):
            if fix not in {"start", "end", "center"}:
                raise ValueError("'fix' must be 'start', 'end', or 'center'")
            fix_arr = np.array([fix] * len(output))
        else:
            fix_arr = np.asarray(fix)
            if not all(f in {"start", "end", "center"} for f in fix_arr):
                raise ValueError("'fix' must contain only 'start', 'end', or 'center'")

        # calculate new starts based on fix point
        new_starts = output._start.copy()
        width_diff = output._width - width_arr.data

        # end fixed
        end_mask = fix_arr == "end"
        new_starts[end_mask] += width_diff[end_mask]

        # center fixed
        center_mask = fix_arr == "center"
        new_starts[center_mask] += width_diff[center_mask] // 2

        output._start = new_starts
        output._width = width_arr.data
        return output

    def flank(self, width: int, start: bool = True, both: bool = False, in_place: bool = False) -> "IRanges":
        """Compute flanking ranges for each range. The logic is from the `IRanges` package.

        If ``start`` is ``True`` for a given range, the flanking occurs at the `start`,
        otherwise the `end`.
        The `widths` of the flanks are given by the ``width`` parameter.

        ``width`` can be negative, in which case the flanking region is
        reversed so that it represents a prefix or suffix of the range.

        Notes:

            `ir.flank(3, True)`, where "x" indicates a range in ``ir`` and "-" indicates the
            resulting flanking region:
                ---xxxxxxx
            If ``start`` were ``False``, the range in ``ir`` becomes
                xxxxxxx---
            For negative width, i.e. `ir.flank(x, -3, FALSE)`, where "*" indicates the overlap
            between "x" and the result:
                xxxx***
            If ``both`` is ``True``, then, for all ranges in "x", the flanking regions are
            extended into (or out of, if ``width`` is negative) the range, so that the result
            straddles the given endpoint and has twice the width given by width.

            This is illustrated below for `ir.flank(3, both=TRUE)`:
                ---***xxxx

            Checkout the documentation on the Bioc package for more details.

        Args:
            width:
                Width to flank by. May be negative.

            start:
                Whether to only flank starts.
                Defaults to True.

            both:
                Whether to flank both starts and ends.
                Defaults to False.

            in_place:
                Whether to modify the object in place.
                Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            flanked ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        output = self._define_output(in_place)

        width_arr = normalize_array(width, len(output))

        if isinstance(start, bool):
            start_arr = np.full(len(output), start, dtype=bool)
        else:
            start_arr = np.asarray(start, dtype=bool)
            if len(start_arr) != len(output):
                start_arr = np.resize(start_arr, len(output))  # may be throw an error?

        if not isinstance(both, bool):
            raise ValueError("'both' must be TRUE or FALSE")

        ends = output.get_end()

        # Handle both-sided flanking
        if both:
            width_abs = np.abs(width_arr)
            new_widths = 2 * width_abs.data
            new_starts = np.where(start_arr, output._start - width_abs.data, ends - width_abs.data + 1)
        else:
            new_widths = np.abs(width_arr.data)

            new_starts = np.zeros(len(output), dtype=np.int32)
            pos_width = width_arr >= 0

            # start=True, width>=0: start - width
            mask1 = start_arr & pos_width
            new_starts[mask1] = output._start[mask1] - width_arr[mask1]

            # start=True, width<0: start
            mask2 = start_arr & ~pos_width
            new_starts[mask2] = output._start[mask2]

            # start=False, width>=0: end + 1
            mask3 = ~start_arr & pos_width
            new_starts[mask3] = ends[mask3] + 1

            # start=False, width<0: end + width + 1
            mask4 = ~start_arr & ~pos_width
            new_starts[mask4] = ends[mask4] + width_arr[mask4] + 1

        output._start = new_starts
        output._width = new_widths
        return output

    def promoters(self, upstream: int = 2000, downstream: int = 200, in_place: bool = False) -> "IRanges":
        """Get promoter regions (upstream and downstream of TSS sites).

        Generates promoter ranges relative to the transcription start site (TSS),
        where TSS is start(x). The promoter range is expanded around the TSS
        according to the upstream and downstream arguments. Upstream represents
        the number of nucleotides in the 5' direction and downstream the number
        in the 3' direction. The full range is defined as, (`start(x) - upstream`)
        to (`start(x) + downstream - 1`).

        Args:
            upstream:
                Number of positions to extend in the 5' direction.
                Defaults to 2000.

            downstream:
                Number of positions to extend in the 3' direction.
                Defaults to 200.

            in_place:
                Whether to modify the object in place. Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            promoter ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        output = self._define_output(in_place)

        new_starts, new_widths = compute_up_down(output._start, output._width, upstream, downstream, "TSS")

        output._start = new_starts
        output._width = new_widths
        return output

    def terminators(self, upstream: int = 2000, downstream: int = 200, in_place: bool = False) -> "IRanges":
        """Get terminator regions (upstream and downstream of TES).

        Args:
            upstream:
                Number of positions to extend in the 5' direction.
                Defaults to 2000.

            downstream:
                Number of positions to extend in the 3' direction.
                Defaults to 200.

            in_place:
                Whether to modify the object in place. Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            terminator ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        output = self._define_output(in_place)

        new_starts, new_widths = compute_up_down(output._start, output._width, upstream, downstream, "TES")

        output._start = new_starts
        output._width = new_widths
        return output

    def reflect(self, bounds: "IRanges", in_place: bool = False) -> "IRanges":
        """Reverses each range in x relative to the corresponding range in bounds.

        Reflection preserves the width of a range, but shifts it such the distance
        from the left bound to the start of the range becomes the distance from the
        end of the range to the right bound. This is illustrated below, where x
        represents a range in x and [ and ] indicate the bounds:

            [..xxx.....]
            becomes
            [.....xxx..]

        Args:
            bounds:
                IRanges with the same length as the current object specifying the bounds.

            in_place:
                Whether to modify the object in place. Defaults to False.

        Returns:
            If ``in_place = False``, a new ``IRanges`` is returned with the
            reflected ranges. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        if not isinstance(bounds, IRanges):
            raise TypeError("'bounds' must be an IRanges object.")

        output = self._define_output(in_place)

        bounds_start = normalize_array(bounds._start, len(output))
        bounds_width = normalize_array(bounds._width, len(output))

        if len(output) > 1 and len(bounds_start) == 0:
            raise ValueError("'bounds' is an empty array")

        ends = output.get_end()
        new_starts = (2 * bounds_start + bounds_width - 1) - ends

        output._start = new_starts
        return output

    def restrict(
        self,
        start: Optional[Union[int, List[int], np.ndarray]] = None,
        end: Optional[Union[int, List[int], np.ndarray]] = None,
        keep_all_ranges: bool = False,
    ) -> "IRanges":
        """Restrict ranges to a given start and end positions.

        Args:
            start:
                Start position. Defaults to None.

            end:
                End position. Defaults to None.

            keep_all_ranges:
                Whether to keep ranges that do not overlap with start and end.
                Defaults to False.

        Returns:
            A new ``IRanges`` with the restricted ranges.
        """
        start_arr = normalize_array(start, len(self))
        end_arr = normalize_array(end, len(self))

        range_ends = self.get_end()

        new_starts = self._start.copy()
        new_ends = range_ends.copy()
        keep_mask = np.ones(len(self), dtype=bool)

        drop_mode = 2 if keep_all_ranges else 1

        # left/start restrictions
        if not start_arr.mask.all():
            if drop_mode == 2:
                # keep but clip ranges
                too_left = (~start_arr.mask) & (new_starts < start_arr)
                new_starts[too_left] = start_arr[too_left]
            else:
                # In drop_mode 1, keep ranges that end at start-1 or later
                far_left = (~start_arr.mask) & (range_ends < start_arr - 1)
                keep_mask &= ~far_left
                # Then adjust remaining starts if needed
                too_left = (~start_arr.mask) & (new_starts < start_arr)
                new_starts[too_left] = start_arr[too_left]

        # right/end restriction
        if not end_arr.mask.all():
            if drop_mode == 2:
                # First handle ranges that are too far right by setting their starts to end+1
                far_right = (~end_arr.mask) & (new_starts > end_arr + 1)
                new_starts[far_right] = end_arr[far_right] + 1
                # Then clip any remaining ends that exceed the boundary
                too_right = (~end_arr.mask) & (new_ends > end_arr)
                new_ends[too_right] = end_arr[too_right]
            else:
                # In drop_mode 1, keep ranges that start at end+1 or earlier
                far_right = (~end_arr.mask) & (new_starts > end_arr + 1)
                keep_mask &= ~far_right
                # Then adjust remaining ends if needed
                too_right = (~end_arr.mask) & (new_ends > end_arr)
                new_ends[too_right] = end_arr[too_right]

        # if keep_all_ranges is True, some ranges may not
        # satisfy the validation requirements
        validate = False
        if drop_mode != 2:
            new_starts = new_starts[keep_mask]
            new_ends = new_ends[keep_mask]
            validate = True

        new_widths = new_ends - new_starts + 1
        return IRanges(
            new_starts, new_widths, mcols=BiocFrame({"revmap": np.where(keep_mask == 1)[0]}), validate=validate
        )

    def threebands(
        self,
        start: Optional[Union[int, np.ndarray]] = None,
        end: Optional[Union[int, np.ndarray]] = None,
        width: Optional[Union[int, np.ndarray]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split ranges into three parts: left, middle, and right.

        Args:
            starts:
                Array of start positions.

            widths:
                Array of widths.

            start:
                Start positions for middle band.

            end:
                End positions for middle band.

            width:
                Width for middle band.

        Returns:
            Dictionary with:
                'left': IRanges for left bands
                'middle': IRanges for middle bands
                'right': IRanges for right bands
        """
        # calculate middle band using narrow
        middle_ranges = self.narrow(start, end, width)
        middle_ends = middle_ranges.get_end()

        # calculate left band
        left_starts = self._start.copy()
        left_widths = middle_ranges._start - self._start

        # calculate right band
        right_starts = middle_ends + 1
        range_ends = self.get_end()
        right_widths = range_ends - middle_ends

        return {
            "left": IRanges(left_starts, left_widths),
            "middle": middle_ranges,
            "right": IRanges(right_starts, right_widths),
        }

    def overlap_indices(self, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """Find overlaps with the start and end positions.

        Args:
            start:
                Start position. Defaults to None.

            end:
                End position. Defaults to None.

        Returns:
            NumPy vector containing indices that overlap with
            the given range.
        """
        counter = 0
        overlaps = []
        for _, val in self:
            keep_s = True
            keep_e = True
            _start = val.start[0]
            _end = val.end[0] - 1

            if start is not None and (_start < start):
                keep_s = False

            if end is not None and (_end > end):
                keep_e = False

            if keep_s is True or keep_e is True:
                overlaps.append(counter)

            counter += 1

        return np.asarray(overlaps)

    ########################
    #### set operations ####
    ########################

    def union(self, other: "IRanges") -> "IRanges":
        """Find union of ranges with `other`.

        Args:
            other:
                An `IRanges` object.

        Raises:
            TypeError:
                If ``other`` is not `IRanges`.

        Returns:
            A new ``IRanges`` object with all ranges.
        """

        if not isinstance(other, IRanges):
            raise TypeError("'other' is not an `IRanges` object.")

        all_starts = combine_sequences(self.start, other.start)
        all_widths = combine_sequences(self.width, other.width)

        output = IRanges(all_starts, all_widths)
        output = output.reduce(min_gap_width=0, drop_empty_ranges=True)
        return output

    def setdiff(self, other: "IRanges") -> "IRanges":
        """Find set difference with `other`.

        Args:
            other:
                An `IRanges` object.

        Raises:
            TypeError:
                If ``other`` is not `IRanges`.

        Returns:
            A new ``IRanges`` object.
        """

        if not isinstance(other, IRanges):
            raise TypeError("'other' is not an `IRanges` object.")

        start = min(self.start.min(), other.start.min())
        end = max(self.end.max(), other.end.max())

        x_gaps = self.gaps(start=start, end=end)
        x_gaps_u = x_gaps.union(other)
        diff = x_gaps_u.gaps(start=start, end=end)

        return diff

    def intersect(self, other: "IRanges") -> "IRanges":
        """Find intersecting ranges with `other`.

        Args:
            other:
                An `IRanges` object.

        Raises:
            TypeError:
                If ``other`` is not `IRanges`.

        Returns:
            A new ``IRanges`` object with all intersecting ranges.
        """

        warn("Use intersect_ncls for a faster implementation.", UserWarning)

        if not isinstance(other, IRanges):
            raise TypeError("'other' is not an `IRanges` object.")

        start = min(self.start.min(), other.start.min())
        end = max(self.end.max(), other.end.max())

        _gaps = other.gaps(start=start, end=end)
        _inter = self.setdiff(_gaps)

        return _inter

    # Inspired by pyranges intersection using NCLS
    # https://github.com/pyranges/pyranges/blob/master/pyranges/methods/intersection.py
    def intersect_ncls(self, other: "IRanges", delete_index: bool = True, num_threads: int = 1) -> "IRanges":
        """Find intersecting ranges with `other`. Uses the nclist index.

        Args:
            other:
                An `IRanges` object.

            delete_index:
                Defaults to True, to delete the cached ncls index.
                Set to False, to reuse the index across multiple queries.

            num_threads:
                Number of threads to use.
                Defaults to 1.

        Raises:
            TypeError:
                If ``other`` is not `IRanges`.

        Returns:
            A new ``IRanges`` object with all intersecting ranges.
        """

        other._build_ncls_index()

        res = other._nclist.find_overlaps(
            self.get_start().astype(np.int32), self.get_end().astype(np.int32) + 1, num_threads=num_threads
        )

        if delete_index:
            other._delete_ncls_index()

        self_new_starts = self.get_start()[list(res[1])]
        other_new_starts = other.get_start()[list(res[0])]

        new_starts = np.where(self_new_starts > other_new_starts, self_new_starts, other_new_starts)

        self_new_ends = self.get_end()[list(res[1])]
        other_new_ends = other.get_end()[list(res[0])]

        new_ends = np.where(self_new_ends < other_new_ends, self_new_ends, other_new_ends)

        return IRanges(new_starts, new_ends - new_starts + 1).reduce()

    ############################
    #### Overlap operations ####
    ############################

    def _build_ncls_index(self):
        if not hasattr(self, "_nclist"):
            self._nclist = libir.NCListHandler(self.get_start().astype(np.int32), self.get_end().astype(np.int32) + 1)

    def _delete_ncls_index(self):
        if hasattr(self, "_nclist"):
            del self._nclist

    def find_overlaps(
        self,
        query: "IRanges",
        query_type: Literal["any", "start", "end", "within"] = "any",
        select: Literal["all", "first", "last", "arbitrary"] = "all",
        max_gap: int = -1,
        min_overlap: int = 0,
        delete_index: bool = True,
        num_threads: int = 1,
    ) -> BiocFrame:
        """Find overlaps with ``query``.

        Args:
            query:
                Query `IRanges`.

            query_type:
                Overlap query type, must be one of

                - "any": Any overlap is good
                - "start": Overlap at the beginning of the range
                - "end": Must overlap at the end of the range
                - "within": Fully contain the query interval

                Defaults to "any".

            select:
                Determine what hit to choose when
                there are multiple hits for a query range.

                Must be one of "all", "first", "last", "arbitrary".

                Defaults to "all".

            max_gap:
                Maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).

            min_overlap:
                Minimum overlap with query.
                Defaults to 1.

            delete_index:
                Defaults to True, to delete the cached ncls index.
                Set to False, to reuse the index across multiple queries.

            num_threads:
                Number of threads to use.
                Defaults to 1.

        Returns:
            A BiocFrame with two columns:
            - query_hits: Indices into query ranges
            - self_hits: Corresponding indices into self ranges that are upstream
            Each row represents a query-subject pair where subject precedes query.
        """

        if max_gap < -1:
            raise ValueError("'max_gap' must be >= -1")

        if not isinstance(query, IRanges):
            raise TypeError("'query' is not a `IRanges` object.")

        if min_overlap < 0:
            raise ValueError("'min_overlap' cannot be negative.")

        if query_type not in ["any", "start", "end", "within"]:
            raise ValueError(f"'query_type' must be one of {', '.join(['any', 'start', 'end', 'within'])}.")

        if select not in ["all", "first", "last", "arbitrary"]:
            raise ValueError(f"'select' must be one of {', '.join(['all', 'first', 'last', 'arbitrary'])}.")

        # R rule: when type="any", at least one of maxgap and minoverlap must be at default
        if query_type == "any" and max_gap != -1 and min_overlap != 0:
            raise ValueError(
                "when query_type='any', at least one of 'max_gap' and 'min_overlap' must be set to its default value"
            )

        if len(query) == 0 or len(self) == 0:
            return BiocFrame(
                data={"self_hits": np.array([], dtype=np.int32), "query_hits": np.array([], dtype=np.int32)}
            )

        if len(self) >= len(query):
            self._build_ncls_index()

            _overlaps = self._nclist.find_overlaps(
                query.get_start().astype(np.int32),
                query.get_end().astype(np.int32) + 1,
                min_overlap=min_overlap,
                max_gap=max_gap,
                query_type=query_type,
                select=select,
                num_threads=num_threads,
            )

            if delete_index:
                self._delete_ncls_index()

            if len(_overlaps) == 0:
                return BiocFrame(
                    data={"self_hits": np.array([], dtype=np.int32), "query_hits": np.array([], dtype=np.int32)}
                )

            return BiocFrame(data={"self_hits": _overlaps[0], "query_hits": _overlaps[1]})
        else:
            query._build_ncls_index()

            _overlaps = query._nclist.find_overlaps(
                self.get_start().astype(np.int32),
                self.get_end().astype(np.int32) + 1,
                min_overlap=min_overlap,
                max_gap=max_gap,
                query_type=query_type,
                select=select,
                num_threads=num_threads,
            )

            if delete_index:
                query._delete_ncls_index()

            if len(_overlaps) == 0:
                return BiocFrame(
                    data={"self_hits": np.array([], dtype=np.int32), "query_hits": np.array([], dtype=np.int32)}
                )

            return BiocFrame(data={"query_hits": _overlaps[0], "self_hits": _overlaps[1]})

    def count_overlaps(
        self,
        query: "IRanges",
        query_type: Literal["any", "start", "end", "within"] = "any",
        max_gap: int = -1,
        min_overlap: int = 0,
        delete_index: bool = True,
        num_threads: int = 1,
    ) -> np.ndarray:
        """Count number of overlaps for each range in ``query``.

        Args:
            query:
                Query `IRanges`.

            query_type:
                Overlap query type, must be one of

                - "any": Any overlap is good
                - "start": Overlap at the beginning of the range
                - "end": Must overlap at the end of the range
                - "within": Fully contain the query interval

                Defaults to "any".

            max_gap:
                Maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).

            min_overlap:
                Minimum overlap with query.
                Defaults to 1.

            delete_index:
                Defaults to True, to delete the cached ncls index.
                Set to False, to reuse the index across multiple queries.

            num_threads:
                Number of threads to use.
                Defaults to 1.

        Returns:
            NumPy vector with length same as number of query ranges,
            value represents the number of overlaps in `self` for each query.
        """
        _overlaps = self.find_overlaps(
            query,
            query_type=query_type,
            max_gap=max_gap,
            min_overlap=min_overlap,
            delete_index=delete_index,
            num_threads=num_threads,
        )
        result = np.zeros(len(query))
        _ucounts = np.unique_counts(_overlaps.get_column("query_hits"))
        result[_ucounts.values] = _ucounts.counts

        return result

    def subset_by_overlaps(
        self,
        query: "IRanges",
        query_type: Literal["any", "start", "end", "within"] = "any",
        select: Literal["all", "first", "last", "arbitrary"] = "all",
        max_gap: int = -1,
        min_overlap: int = 0,
        delete_index: bool = True,
        num_threads: int = 1,
    ) -> "IRanges":
        """Subset to overlapping ranges with ``query``.

        Args:
            query:
                Query ``IRanges`` object.

            query_type:
                Overlap query type, must be one of

                - "any": Any overlap is good
                - "start": Overlap at the beginning of the range
                - "end": Must overlap at the end of the range
                - "within": Fully contain the query interval

                Defaults to "any".

            select:
                Determine what hit to choose when
                there are multiple hits for a query range.

                Must be one of "all", "first", "last", "arbitrary".

                Defaults to "all".

            max_gap:
                Maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).

            min_overlap:
                Minimum overlap with query.
                Defaults to 1.

            delete_index:
                Defaults to True, to delete the cached ncls index.
                Set to False, to reuse the index across multiple queries.

            num_threads:
                Number of threads to use.
                Defaults to 1.

        Returns:
            A new ``IRanges`` object containing ranges that overlap with query.
        """
        _overlaps = self.find_overlaps(
            query=query,
            query_type=query_type,
            select=select,
            max_gap=max_gap,
            min_overlap=min_overlap,
            delete_index=delete_index,
            num_threads=num_threads,
        )
        _all_indices = np.unique(_overlaps.get_column("self_hits"))
        return self[_all_indices]

    ###########################
    #### search operations ####
    ###########################

    def _build_nclssearch_index(self):
        if not hasattr(self, "_nclistsearch"):
            self._nclistsearch = libir.NCListSearchHandler(
                self.get_start().astype(np.int32), self.get_end().astype(np.int32) + 1
            )

    def _delete_nclssearch_index(self):
        if hasattr(self, "_nclistsearch"):
            del self._nclistsearch

    def precede(
        self,
        query: "IRanges",
        select: Literal["all", "first"] = "first",
        delete_index: bool = True,
        num_threads: int = 1,
    ) -> Union[np.ndarray, BiocFrame]:
        """Find nearest positions that are upstream/precede each query range.

        Args:
            query:
                Query `IRanges`.

            select:
                Whether to return "all" hits or just "first".
                Defaults to "first".

            delete_index:
                Defaults to True, to delete the cached ncls index.
                Set to False, to reuse the index across multiple queries.

            num_threads:
                Number of threads to use.
                Defaults to 1.

        Returns:
            If select="first":
                A numpy array of integers with length matching query, containing indices
                into self for the closest upstream position of each query range. Value may be
                None if there are no matches.
            If select="all":
                A BiocFrame with two columns:
                - query_hits: Indices into query ranges
                - self_hits: Corresponding indices into self ranges that are upstream
                Each row represents a query-self pair where self precedes query.
        """

        if not isinstance(query, IRanges):
            raise TypeError("`query` is not a `IRanges` object.")

        if select not in ["all", "first"]:
            raise ValueError(f"'select' must be one of {', '.join(['all', 'first'])}.")

        self._build_nclssearch_index()

        _results = self._nclistsearch.precede(
            query.get_end().astype(np.int32) + 1,
            select=select,
            num_threads=num_threads,
        )

        if delete_index:
            self._delete_nclssearch_index()

        if select == "first":
            _results = np.asarray(_results, dtype=np.object_)
            # replace -1 with None
            _results[_results == -1] = None
            return _results
        else:
            return BiocFrame(data={"query_hits": _results[0], "self_hits": _results[1]})

    def follow(
        self,
        query: "IRanges",
        select: Literal["all", "last"] = "last",
        delete_index: bool = True,
        num_threads: int = 1,
    ) -> Union[np.ndarray, BiocFrame]:
        """Find nearest positions that are downstream/follow each query range.

        Args:
            query:
                Query `IRanges`.

            select:
                Whether to return "all" hits or just "last".
                Defaults to "last".

            delete_index:
                Defaults to True, to delete the cached ncls index.
                Set to False, to reuse the index across multiple queries.

            num_threads:
                Number of threads to use.
                Defaults to 1.

        Returns:
            If select="last":
                A numpy array of integers with length matching query, containing indices
                into self for the closest downstream position of each query range. Value may be
                None if there are no matches.
            If select="all":
                A BiocFrame with two columns:
                - query_hits: Indices into query ranges
                - self_hits: Corresponding indices into self ranges that are upstream
                Each row represents a query-self pair where self follows query.
        """

        if not isinstance(query, IRanges):
            raise TypeError("`query` is not a `IRanges` object.")

        if select not in ["all", "last"]:
            raise ValueError(f"'select' must be one of {', '.join(['all', 'last'])}.")

        self._build_nclssearch_index()

        _results = self._nclistsearch.follow(
            query.get_start().astype(np.int32),
            select=select,
            num_threads=num_threads,
        )

        if delete_index:
            self._delete_nclssearch_index()

        if select == "last":
            _results = np.asarray(_results, dtype=np.object_)
            # replace -1 with None
            _results[_results == -1] = None
            return _results
        else:
            return BiocFrame(data={"query_hits": _results[0], "self_hits": _results[1]})

    def distance(self, query: "IRanges") -> np.ndarray:
        """Calculate the pair-wise distance between ranges.

        Args:
            query:
                Query `IRanges`.

        Returns:
            NumPy vector containing distances for each range in query.
        """
        if not isinstance(query, IRanges):
            raise TypeError("`query` is not a `IRanges` object.")

        if len(self) != len(query):
            raise ValueError("'query' does not contain the same number of ranges.")

        max_starts = np.maximum(self._start, query._start)
        min_ends = np.minimum(self.get_end(), query.get_end())
        return np.maximum(max_starts - min_ends - 1, 0)

    def nearest(
        self,
        query: "IRanges",
        select: Literal["all", "arbitrary"] = "arbitrary",
        delete_index: bool = True,
        num_threads: int = 1,
    ) -> Union[np.ndarray, BiocFrame]:
        """Find nearest ranges in both directions.

        Args:
            query:
                Query IRanges.

            select:
                Whether to return "all" hits or "arbitrary" choice.

            delete_index:
                Delete the cached ncls index.
                Internal use only.

            num_threads:
                Number of threads to use.
                Defaults to 1.

        Returns:
            If select="arbitrary":
                A numpy array of integers with length matching query, containing indices
                into self for the closest for each query range. Value may be None if there
                are no matches.
            If select="all":
                A BiocFrame with two columns:
                - query_hits: Indices into query ranges
                - self_hits: Corresponding indices into self ranges that are upstream
                Each row represents a query-subject pair where subject is nearest to query.
        """
        if not isinstance(query, IRanges):
            raise TypeError("`query` is not a `IRanges` object.")

        if select not in ["all", "arbitrary"]:
            raise ValueError("'select' must be one of 'all', 'arbitrary'")

        self._build_nclssearch_index()

        _results = self._nclistsearch.nearest(
            query.get_start().astype(np.int32),
            query.get_end().astype(np.int32) + 1,
            select=select,
            num_threads=num_threads,
        )

        if delete_index:
            self._delete_nclssearch_index()

        if select == "arbitrary":
            _results = np.asarray(_results, dtype=np.object_)
            # replace -1 with None
            _results[_results == -1] = None
            return _results
        else:
            return BiocFrame(data={"query_hits": _results[0], "self_hits": _results[1]})

    ########################
    #### pandas interop ####
    ########################

    def to_pandas(self):
        """Convert this IRanges object to a :py:class:`~pandas.DataFrame`.

        Returns:
            A :py:class:`~pandas.DataFrame` object.
        """
        import pandas as pd

        _starts = self._start
        _widths = self._width
        _ends = self.get_end()

        output = pd.DataFrame({"starts": _starts, "widths": _widths, "ends": _ends})

        if self._mcols is not None and self._mcols.shape[1] > 0:
            output = pd.concat([output, self._mcols.to_pandas()], axis=1)

        if self._names is not None:
            output.index = self._names

        return output

    @classmethod
    def from_pandas(cls, input) -> "IRanges":
        """Create an ``IRanges`` object from a :py:class:`~pandas.DataFrame`.

        Args:
            input:
                Input data must contain columns 'start' and 'width'.

        Returns:
            A ``IRanges`` object.
        """

        from pandas import DataFrame

        if not isinstance(input, DataFrame):
            raise TypeError("`input` is not a pandas `DataFrame` object.")

        if "start" not in input.columns:
            raise ValueError("'input' must contain column 'start'.")
        start = input["start"].tolist()

        if "width" not in input.columns:
            raise ValueError("'input' must contain column 'width'.")
        width = input["width"].tolist()

        # mcols
        mcols_df = input.drop(columns=["start", "width"])

        mcols = None
        if (not mcols_df.empty) or len(mcols_df.columns) > 0:
            mcols = BiocFrame.from_pandas(mcols_df)

        names = None
        if input.index is not None:
            names = [str(i) for i in input.index.to_list()]

        return cls(start=start, width=width, names=names, mcols=mcols)

    ########################
    #### polars interop ####
    ########################

    def to_polars(self):
        """Convert this ``IRanges`` object to a :py:class:`~polars.DataFrame`.

        Returns:
            A :py:class:`~polars.DataFrame` object.
        """
        import polars as pl

        _starts = self._start
        _widths = self._width
        _ends = self.get_end()

        output = pl.DataFrame({"starts": _starts, "widths": _widths, "ends": _ends})

        if self._mcols is not None and self._mcols.shape[1] > 0:
            output = pl.concat([output, self._mcols.to_polars()], how="horizontal")

        if self._names is not None:
            output = output.with_columns(names=self._names)

        return output

    @classmethod
    def from_polars(cls, input) -> "IRanges":
        """Create an ``IRanges`` object from a :py:class:`~polars.DataFrame`.

        Args:
            input:
                Input data must contain columns 'start' and 'width'.

        Returns:
            A ``IRanges`` object.
        """

        from polars import DataFrame

        if not isinstance(input, DataFrame):
            raise TypeError("`input` is not a polars `DataFrame` object.")

        if "start" not in input.columns:
            raise ValueError("'input' must contain column 'start'.")
        start = input["start"].to_list()

        if "width" not in input.columns:
            raise ValueError("'input' must contain column 'width'.")
        width = input["width"].to_list()

        # mcols
        mcols_df = input.drop(["start", "width"])

        mcols = None
        if (not mcols_df.is_empty()) or len(mcols_df.columns) > 0:
            mcols = BiocFrame.from_polars(mcols_df)

        names = None

        return cls(start=start, width=width, names=names, mcols=mcols)

    ##############
    #### misc ####
    ##############

    @classmethod
    def empty(cls):
        """Create an zero-length ``IRanges`` object.

        Returns:
            Same type as caller, in this case a ``IRanges``.
        """
        return cls([], [])

    #############################
    #### combine ops wrapper ####
    #############################

    def combine(self, *other: "IRanges") -> "IRanges":
        """Combine multiple range objects into one.

        Wrapper around :py:func:`~biocutils.combine_sequences`.

        Returns:
            An `IRanges` containing all the combined ranges.
        """
        return _combine_IRanges(self, *other)

    ################################
    ######>> window methods <<######
    ################################

    def tile(
        self, n: Optional[Union[int, np.ndarray]] = None, width: Optional[Union[int, np.ndarray]] = None
    ) -> List["IRanges"]:
        """Split ranges into either n equal parts or parts of fixed width.

        Args:
            n:
                Number of tiles per range (mutually exclusive with width).

            width:
                Width of each tile (mutually exclusive with n).

        Returns:
            List of `IRanges` objects, one per input range containing the tiles.
        """
        if n is not None and width is not None:
            raise ValueError("only one of 'n' and 'width' can be specified")

        if n is not None:
            if isinstance(n, int):
                n = np.repeat(n, len(self))
            else:
                n = np.asarray(n, dtype=np.int32)

            if np.any(self._width < n):
                raise ValueError("some width(x) are less than 'n'")
            if np.any(n < 0):
                raise ValueError("some 'n' are negative")

            # tile_widths = self._width / n

        if width is not None:
            if isinstance(n, int):
                width = np.repeat(width, len(self))
            else:
                width = np.asarray(width, dtype=np.int32)

            if np.any(width < 0):
                raise ValueError("some 'width' are negative")

            n = np.ceil(self._width / width).astype(np.int32)
            # tile_widths = np.repeat(width, len(self))

        widths_per_tile = self._width / n
        positions = np.arange(1, np.max(n) + 1, dtype=np.int32)

        result = []
        for i in range(len(self)):
            rel_ends = np.floor(positions[: n[i]] * widths_per_tile[i]).astype(np.int32)
            abs_ends = rel_ends + self._start[i] - 1
            prev_ends = np.concatenate([[self._start[i] - 1], abs_ends[:-1]])
            widths = abs_ends - prev_ends
            result.append(IRanges(start=abs_ends - widths + 1, width=widths))

        return result

    def sliding_windows(self, width: int, step: int = 1) -> List["IRanges"]:
        """Create sliding windows of fixed width and step size.

        Args:
            width:
                Width of each window.

            step:
                Step size between window starts.

        Returns:
            List of `IRanges` objects, one per input range containing the windows.
        """
        if not isinstance(width, (int, np.integer)):
            raise ValueError("'width' must be a single, non-NA number")

        if not isinstance(step, (int, np.integer)):
            raise ValueError("'step' must be a single, non-NA number")

        if width < 0:
            raise ValueError("'width' cannot be negative")

        if step < 0:
            raise ValueError("'step' cannot be negative")

        n = np.ceil(np.maximum(self._width - width, 0) / step).astype(np.int32) + 1

        result = []
        for i in range(len(self)):
            window_starts = np.arange(n[i]) * step + 1

            valid_windows = window_starts + width - 1 <= self._width[i]
            if not np.all(valid_windows):
                window_starts = window_starts[valid_windows]

            abs_starts = window_starts + self._start[i] - 1
            window_widths = np.repeat(width, len(window_starts))

            result.append(IRanges(start=abs_starts, width=window_widths))

        return result


@combine_sequences.register
def _combine_IRanges(*x: IRanges) -> IRanges:
    has_names = False
    for y in x:
        if y._names is not None:
            has_names = True
            break

    all_names = None
    if has_names:
        all_names = []
        for y in x:
            if y._names is not None:
                all_names += y._names
            else:
                all_names += [""] * len(y)

    return IRanges(
        start=combine_sequences(*[y._start for y in x]),
        width=combine_sequences(*[y._width for y in x]),
        names=all_names,
        mcols=combine_rows(*[y._mcols for y in x]),
        metadata=x[0]._metadata,
        validate=False,
    )
