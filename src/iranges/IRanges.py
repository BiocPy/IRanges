from copy import deepcopy
from typing import List, Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import biocutils as ut
from biocframe import BiocFrame
from biocgenerics import combine_rows, combine_seqs, show_as_cell
from numpy import array, clip, int32, ndarray, printoptions, where, zeros

from .interval import create_np_interval_vector

__author__ = "Aaron Lun, Jayaram Kancherla"
__copyright__ = "LTLA, jkanche"
__license__ = "MIT"


class IRangesIter:
    """An iterator to :py:class:`~iranges.IRanges.IRanges`.

    Args:
        obj (IRanges): Object to iterate.
    """

    def __init__(self, obj: "IRanges") -> None:
        """Initialize the iterator.

        Args:
            obj (IRanges): Source object to iterate.
        """
        self._iranges = obj
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index < len(self._iranges):
            iter_row_index = (
                self._iranges.names[self._current_index]
                if self._iranges.names is not None
                else None
            )

            iter_slice = self._iranges.get_row(self._current_index)
            self._current_index += 1
            return (iter_row_index, iter_slice)

        raise StopIteration


class IRanges:
    """A collection of integer ranges, equivalent to the ``IRanges`` class from the
    `Bioconductor package <https://bioconductor.org/packages/IRanges>`_ of the same name.

    This holds a **start** position and a **width**, and is most typically used to represent coordinates along some genomic
    sequence. The interpretation of the start position depends on the application; for sequences, the start is usually a
    1-based position, but other use cases may allow zero or even negative values.
    """

    def __init__(
        self,
        start: Sequence[int],
        width: Sequence[int],
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
        if isinstance(start, ndarray) and start.dtype == int32:
            return start

        return array(start, dtype=int32, copy=False)

    def _sanitize_width(self, width):
        if isinstance(width, ndarray) and width.dtype == int32:
            return width

        return array(width, dtype=int32, copy=False)

    def _validate_width(self):
        if len(self._start) != len(self._width):
            raise ValueError("'start' and 'width' should have the same length")

        if (self._width < 0).any():
            raise ValueError("'width' must be non-negative")

        if (self._start + self._width < self._start).any():
            raise ValueError("end position should fit in a 32-bit signed integer")

    def _sanitize_names(self, names):
        if names is None:
            return None
        elif not isinstance(names, list):
            names = list(names)

        return names

    def _validate_names(self):
        if self._names is None:
            return None

        if not ut.is_list_of_type(self._names, str):
            raise ValueError("'names' should be a list of strings")

        if len(self._names) != len(self._start):
            raise ValueError("'names' and 'start' should have the same length")

    def _sanitize_mcols(self, mcols):
        if mcols is None:
            return BiocFrame({}, number_of_rows=len(self._start))
        else:
            return mcols

    def _validate_mcols(self):
        if not isinstance(self._mcols, BiocFrame):
            raise TypeError("'mcols' should be a BiocFrame")

        if self._mcols.shape[0] != len(self._start):
            raise ValueError(
                "Number of rows in 'mcols' should be equal to length of 'start'"
            )

    def _sanitize_metadata(self, metadata):
        if metadata is None:
            return {}
        elif not isinstance(metadata, dict):
            metadata = dict(metadata)

        return metadata

    def _validate_metadata(self):
        if not isinstance(self._metadata, dict):
            raise TypeError("'metadata' should be a dictionary")

    ########################
    #### Getter/setters ####
    ########################

    def get_start(self) -> ndarray:
        """Get all start positions.

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
    def start(self) -> ndarray:
        """Get all start positions.

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

    def get_width(self) -> ndarray:
        """Get width of each interval.

        Returns:
            ndarray: NumPy array of 32-bit signed integers containing the widths for all
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
        output._width = output._sanitize_width(width)
        output._validate_width()
        return output

    @property
    def width(self) -> ndarray:
        """Get width of each interval.

        Returns:
            ndarray: NumPy array of 32-bit signed integers containing the widths for all
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

    def get_end(self) -> ndarray:
        """Get all end positions.

        Returns:
            NumPy array of 32-bit signed integers containing the end position
            (not inclusive) for all ranges.
        """
        return self._start + self._width

    @property
    def end(self) -> ndarray:
        """Get all end positions (read-only).

        Returns:
            NumPy array of 32-bit signed integers containing the end position
            (not inclusive) for all ranges.
        """
        return self.get_end()

    def get_names(self) -> Union[None, List[str]]:
        """Get all names.

        Returns:
            List containing the names for all ranges, or None if no names are
            present.
        """
        return self._names

    def set_names(
        self, names: Optional[Sequence[str]], in_place: bool = False
    ) -> "IRanges":
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
    def names(self) -> Union[None, List[str]]:
        """Get all names.

        Returns:
            List containing the names for all ranges, or None if no names are
            present.
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

    def set_mcols(
        self, mcols: Optional[BiocFrame], in_place: bool = False
    ) -> "IRanges":
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
        """Get metadata about ranges.

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

    def set_metadata(
        self, metadata: Optional[dict], in_place: bool = False
    ) -> "IRanges":
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

    def _define_output(self, in_place):
        if in_place:
            return self
        else:
            return self.__copy__()

    #########################
    #### Getitem/setitem ####
    #########################

    def __len__(self) -> int:
        """
        Returns:
            Length of this object.
        """
        return len(self._start)

    def __getitem__(
        self, subset: Union[Sequence, int, str, bool, slice, range]
    ) -> "IRanges":
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
            start=self._start[idx],
            width=self._width[idx],
            names=ut.subset(self._names, idx) if self._names is not None else None,
            mcols=self._mcols[idx, :],
            metadata=self._metadata,
        )

    def __setitem__(
        self, args: Union[Sequence, int, str, bool, slice, range], value: "IRanges"
    ):
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

    ##################
    #### Printing ####
    ##################

    def __repr__(self) -> str:
        with printoptions(threshold=50, edgeitems=3):
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
            sub_end = sub_start + sub_width
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
                    showed = ut.truncate_strings(
                        showed, width=max(40, len(header[0]), len(header[1]))
                    )
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

    def get_row(self, index_or_name: Union[str, int]) -> "IRanges":
        """Access a row by index or row name.

        Args:
            index_or_name (Union[str, int]): Integer index of the row to access.

                Alternatively, you may provide a string specifying the row name to access,
                only if :py:attr:`~iranges.IRanges.IRanges.names` are available.

        Raises:
            ValueError:
                If ``index_or_name`` is not in row names.
                If the integer index is greater than the number of rows.
            TypeError:
                If ``index_or_name`` is neither a string nor an integer.

        Returns:
            IRanges: A sliced IRanges object
        """

        if not isinstance(index_or_name, (int, str)):
            raise TypeError("`index_or_name` must be either an integer index or name.")

        return self[index_or_name]

    def __iter__(self) -> IRangesIter:
        """Iterator over intervals."""
        return IRangesIter(self)

    ############################
    #### iter range methods ####
    ############################

    def clip_intervals(
        self, shift: int = 0, width: Optional[Union[int, List[int]]] = None
    ) -> "IRanges":
        """Clip intervals. Starts are always clipped to positive interval ranges (1, Inf).

        If ``width`` is specified, the intervals are clipped to (1, width).

        Args:
            shift (int, optional): Shift all starts before clipping. Defaults to 0.
            width (Union[int, List[int]], optional): Clip width of each interval. Defaults to None.

        Returns:
            IRanges: A ``IRanges`` object, with the clipped intervals.
        """

        _clipped_starts = []
        _clipped_widths = []
        _clipped_names = []

        counter = 0
        for name, val in self:
            _start = val.start[0]
            _width = val.width[0]

            if shift > 0:
                _start += shift

            if width is not None:
                if not isinstance(width, int):
                    raise TypeError("'width' must be either an integer or a vector.")

                if _start + _width > width:
                    _width = width - _start

            counter += 1

            if _start < 1:
                _start = 1
                _width = val.end[0] - _start

            _end = _start + _width
            if _end < 1:
                continue

            _clipped_starts.append(_start)
            _clipped_widths.append(_width)
            _clipped_names.append(name if name is not None else str(counter - 1))

        if all(x is None for x in _clipped_names):
            _clipped_names = None

        return IRanges(_clipped_starts, _clipped_widths, names=_clipped_names)

    def coverage(
        self, shift: int = 0, width: Optional[int] = None, weight: Union[int, float] = 1
    ) -> ndarray:
        """Calculate coverage, for each position, counts the number of intervals that cover it.

        Args:
            shift (int, optional): Shift all intervals. Defaults to 0.
            width (int, optional): Restrict the width of all intervals. Defaults to None.
            weight (Union[int, float], optional): Weight to use. Defaults to 1.

        Raises:
            - If 'weight' is not a number.
            - If 'width' is not a number.

        Returns:
            ndarray:  A numpy array with the coverage vector.
        """

        new_ranges = self.clip_intervals(shift=shift, width=width)

        if weight is not None and not isinstance(weight, (int, float)):
            raise TypeError("'width' must be an integer or float.")

        cov, _ = create_np_interval_vector(new_ranges, force_size=width, value=weight)
        return cov

    def range(self) -> "IRanges":
        """Concatenate all intervals. A tuple of minimum of all starts, maximum of all ends) in the object.

        Args:
            with_reverse_map (bool, optional): return map of indices back to
                original object?. Defaults to False.

        Returns:
            IRanges: An new IRanges instance with a single range,
            from the minimum start to the maximum end of the concatenated object.
        """

        min_start = min(self.start)
        max_end = max(self.end)

        return IRanges([min_start], [max_end - min_start])

    def reduce(
        self,
        with_reverse_map: bool = False,
        drop_empty_ranges: bool = False,
        min_gap_width: int = 1,
    ) -> "IRanges":
        """`Reduce` orders the ranges, then merges overlapping or adjacent ranges.

        Args:
            with_reverse_map (bool, optional): Whether to return map of indices back to
                original object. Defaults to False.
            drop_empty_ranges (bool, optional): Whether to drop empty ranges. Defaults to False.
            min_gap_width (int, optional): Ranges separated by a gap of at least ``min_gap_width``
                positions are not merged. Defaults to 1.

        Returns:
            IRanges: A new ``IRanges`` object with reduced intervals.
        """
        if min_gap_width < 0:
            raise ValueError("'min_gap_width' cannot be negative.")

        _order = self.order()

        result_starts = []
        result_widths = []
        result_revmaps = []
        counter = 0

        def get_elem_counter(idx):
            elem = self[idx]
            start = elem.start[0]
            end = elem.end[0] - 1
            width = elem.width[0]
            return start, end, width

        current_start, current_end, _ = get_elem_counter(_order[counter])
        current_revmaps = [_order[counter]]

        counter += 1
        while counter < len(_order):
            merge = False

            o = _order[counter]
            _idx_start, _idx_end, _idx_width = get_elem_counter(o)
            _gap_width = _idx_start - current_end

            if _gap_width <= min_gap_width:
                merge = True

            if merge is True:
                if current_end < _idx_end:
                    current_end = _idx_end

                current_revmaps.append(o)
                counter += 1
            else:
                if not (
                    drop_empty_ranges is True and current_end - current_start + 1 == 0
                ):
                    result_starts.append(current_start)
                    result_widths.append(current_end - current_start + 1)
                    result_revmaps.append(current_revmaps)

                current_revmaps = [o]
                current_start = _idx_start
                current_end = _idx_end
                counter += 1

        result_starts.append(current_start)
        result_widths.append(current_end - current_start + 1)
        result_revmaps.append(current_revmaps)

        result = IRanges(result_starts, result_widths)

        if with_reverse_map is True:
            result.set_mcols(BiocFrame({"revmap": result_revmaps}), in_place=True)

        return result

    def _get_intervals_as_list(self) -> List[Tuple[int, int, int]]:
        """Internal method to get intervals as a list of tuples.

        Returns:
            List[Tuple[int, int, int]]: List of tuples containing the start, end and the index.
        """
        intvals = []
        for i in range(len(self)):
            intvals.append((self.start[i], self.end[i], i))

        return intvals

    def order(self, decreasing: bool = False) -> List[int]:
        """Get the order of indices for sorting.

        Args:
            decreasing (bool, optional): Whether to sort in descending order. Defaults to False.

        Returns:
            List[int]: List of integers indicating index position.
        """
        intvals = sorted(self._get_intervals_as_list(), reverse=decreasing)
        order = [o[2] for o in intvals]
        return order

    def sort(self, decreasing: bool = False, in_place: bool = False) -> "IRanges":
        """Sort the intervals.

        Args:
            decreasing (bool): Whether to sort in descending order. Defaults to False.
            in_place (bool): Whether to modify the object in place. Defaults to False.

        Returns:
            IRanges: If ``in_place = False``, a new ``IRanges`` is returned with the
            sorted intervals. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        order = self.order(decreasing=decreasing)
        output = self._define_output(in_place)
        return output[order]

    def gaps(self, start: Optional[int] = None, end: Optional[int] = None) -> "IRanges":
        """Gaps returns an ``IRanges`` object representing the set of integers that remain after the intervals are
        removed specified by the start and end arguments.

        Args:
            start (int, optional): Restrict start position. Defaults to 1.
            end (int, optional): Restrict end position. Defaults to None.

        Returns:
            IRanges: A new ``IRanges`` is returned with the gaps.
        """
        _order = self.order()

        overlap_start = min(self.start) if start is None else start
        overlap_end = overlap_start - 1 if start is not None else None

        result_starts = []
        result_widths = []

        def get_elem_counter(idx):
            elem = self[idx]
            start = elem.start[0]
            end = elem.end[0] - 1
            width = elem.width[0]
            return start, end, width

        for i in range(len(_order)):
            _start, _end, _width = get_elem_counter(_order[i])

            if _width == 0:
                continue

            if overlap_end is None:
                overlap_end = _end
            else:
                _gap_start = overlap_end + 1

                if end is not None and _start > end + 1:
                    _start = end + 1

                _gap_width = _start - _gap_start

                if _gap_width >= 1:
                    result_starts.append(_gap_start)
                    result_widths.append(_gap_width)
                    overlap_end = _end
                elif _end > overlap_end:
                    overlap_end = _end

            if end is not None and overlap_end >= end:
                break

        if end is not None and overlap_end is not None and overlap_end < end:
            result_starts.append(overlap_end + 1)
            result_widths.append(end - overlap_end)

        return IRanges(result_starts, result_widths)

    # folows the same logic as in https://stackoverflow.com/questions/55480499/split-set-of-intervals-into-minimal-set-of-disjoint-intervals
    # otherwise too much magic happening here - https://github.com/Bioconductor/IRanges/blob/5acb46b3f2805f7f74fe4cb746780e75f8257a83/R/inter-range-methods.R#L389
    def disjoin(self, with_reverse_map: bool = False) -> "IRanges":
        """Calculate disjoint intervals.

        Args:
            with_reverse_map (bool, optional): Whether to return a map of indices back to the original object.
                Defaults to False.

        Returns:
            IRanges: A new `IRanges` containing disjoint intervals.
        """
        all_ints = []
        counter = 0
        for _, val in self:
            all_ints.append((val.start[0], 1, counter))
            all_ints.append((val.end[0], -1, counter))

            counter += 1

        sorted_ints = sorted(all_ints)

        counter = 0
        _current_start = None

        result_starts = []
        result_widths = []
        result_revmaps = []

        _curr_revmap = []
        for x in sorted_ints:
            _curr_revmap.append(x[2])

            if _current_start is not None and x[0] > _current_start and counter != 0:
                result_starts.append(_current_start)
                result_widths.append(x[0] - _current_start)
                result_revmaps.append(list(set(_curr_revmap)))
                _curr_revmap = []

            _current_start = x[0]
            counter += x[1]

        result = IRanges(result_starts, result_widths)

        if with_reverse_map is True:
            result.set_mcols(BiocFrame({"revmap": result_revmaps}), in_place=True)

        return result

    #############################
    #### intra range methods ####
    #############################

    def shift(self, shift: int, in_place: bool = False) -> "IRanges":
        """Shifts all the intervals by the amount specified by the ``shift`` argument.

        Args:
            shift (int): Amount to shift by.
            in_place (bool): Whether to modify the object in place. Defaults to False.

        Returns:
            IRanges: If ``in_place = False``, a new ``IRanges`` is returned with the
            shifted intervals. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """
        if not isinstance(shift, int):
            raise TypeError("'shift' must be an integer.")

        output = self._define_output(in_place)
        output._start = output._start + shift
        return output

    # TODO: not fully implemented
    def narrow(
        self,
        start: Optional[int] = None,
        width: Optional[int] = None,
        end: Optional[int] = None,
        in_place: bool = False,
    ) -> "IRanges":
        """Narrow genomic positions by provided ``start``, ``width`` and ``end`` parameters.

        Important: These arguments are relative shift in positions for each range.

        Args:
            start (int, optional): Relative start position. Defaults to None.
            width (int, optional): Relative end position. Defaults to None.
            end (int, optional): Relative width of the interval. Defaults to None.
            in_place (bool): Whether to modify the object in place. Defaults to False.

        Raises:
            ValueError: If `width` is provided, either `start` or `end` must be provided.
            ValueError: Provide two of the three parameters - `start`, `end` and `width`
                but not all.

        Returns:
            IRanges: If ``in_place = False``, a new ``IRanges`` is returned with the
            narrow intervals. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        if start is not None and end is not None and width is not None:
            raise ValueError(
                "Only provide two of the three parameters - 'start', "
                "'end' and 'width' but not all!"
            )

        if width is not None:
            if start is None and end is None:
                raise ValueError(
                    "If 'width' is provided, either 'start' or 'end' must be provided."
                )

        output = self._define_output(in_place)

        all_starts = output.start.copy()
        all_widths = output.width.copy()
        all_ends = output.end.copy()

        if start is not None:
            if start > 0:
                all_starts = all_starts + start - 1
                all_widths = all_widths - start + 1
            else:
                all_starts = all_ends + start
                all_widths = zeros(len(all_widths)) + start

            if width is not None:
                all_widths = zeros(len(all_widths)) + width
            elif end is not None:
                all_widths = all_widths + end + 1
        elif end is not None:
            if end < 0:
                all_widths = all_widths + end - 1
            else:
                all_widths = zeros(len(all_widths)) + end

            if width is not None:
                all_widths = zeros(len(all_widths)) + width
            elif end is not None:
                all_widths = all_widths + end + 1

        output._start = all_starts
        output._width = all_widths
        return output

    def resize(
        self,
        width: int,
        fix: Literal["start", "end", "center"] = "start",
        in_place: bool = False,
    ) -> "IRanges":
        """Resize ranges to the specified ``width`` where either the ``start``, ``end``, or ``center`` is used as an
        anchor.

        Args:
            width (int): Width to resize, must be non-negative!
            fix (Literal["start", "end", "center"], optional): Fix positions by "start", "end", or "center".
                Defaults to "start".
            in_place (bool): Whether to modify the object in place. Defaults to False.

        Raises:
            ValueError: If parameter ``fix`` is neither `start`, `end`, nor `center`.
            ValueError: If ``width`` is negative.

        Returns:
            IRanges: If ``in_place = False``, a new ``IRanges`` is returned with the
            resized intervals. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        if width < 0:
            raise ValueError("`width` cannot be negative!")

        if fix not in ["start", "end", "center"]:
            raise ValueError(
                f"`fix` must be either 'start', 'end' or 'center', provided {fix}"
            )

        output = self._define_output(in_place)
        if fix != "start":
            if fix == "end":
                output._start = output.start + output.width - width

            if fix == "center":
                output._start = output.start + output.width - (width / 2)

        output._width = width
        return output

    def flank(
        self, width: int, start: bool = True, both: bool = False, in_place: bool = False
    ) -> "IRanges":
        """Compute flanking ranges for each range. The logic is from the `IRanges` package.

        If ``start`` is ``True`` for a given range, the flanking occurs at the `start`,
        otherwise the `end`.
        The `widths` of the flanks are given by the ``width`` parameter.

        ``width`` can be negative, in which case the flanking region is
        reversed so that it represents a prefix or suffix of the range.

        Usage:

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

        Args:
            width (int): Width to flank by. May be negative.
            start (bool, optional): Whether to only flank starts. Defaults to True.
            both (bool, optional): Whether to flank both starts and ends. Defaults to False.
            in_place (bool): Whether to modify the object in place. Defaults to False.

        Returns:
            IRanges: If ``in_place = False``, a new ``IRanges`` is returned with the
            flanked intervals. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        output = self._define_output(in_place)

        if both is True:
            width = abs(width)
            if start is True:
                output._start = output.start - width
            else:
                output._start = output.end - width

            output._width = zeros(len(output)) + (2 * width)
        else:
            if start is True:
                if width >= 0:
                    output._start = output.start - width
            else:
                if width >= 0:
                    output._start = output.end
                else:
                    output._start = output.end + width

            output._width = zeros(len(output)) + abs(width)

        return output

    def promoters(
        self, upstream: int = 2000, downstream: int = 200, in_place: bool = False
    ) -> "IRanges":
        """Extend intervals to promoter regions.

        Generates promoter ranges relative to the transcription start site (TSS),
        where TSS is start(x). The promoter range is expanded around the TSS
        according to the upstream and downstream arguments. Upstream represents
        the number of nucleotides in the 5' direction and downstream the number
        in the 3' direction. The full range is defined as, (`start(x) - upstream`)
        to (`start(x) + downstream - 1`).

        Args:
            upstream (int, optional): Number of positions to extend in the 5' direction.
                Defaults to 2000.
            downstream (int, optional): Number of positions to extend in the 3' direction.
                Defaults to 200.
            in_place (bool): Whether to modify the object in place. Defaults to False.

        Returns:
            IRanges: If ``in_place = False``, a new ``IRanges`` is returned with the
            promoter intervals. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        output = self._define_output(in_place)

        if upstream < 0 or downstream < 0:
            raise ValueError("'upstream' and 'downstream; must be integers >=0.")

        new_starts = output.start - upstream
        new_ends = output.start + downstream

        output._start = new_starts
        output._width = new_ends - new_starts
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
            bounds (IRanges): IRanges with the same length as the current object specifying the bounds.
            in_place (bool): Whether to modify the object in place. Defaults to False.

        Returns:
            IRanges: If ``in_place = False``, a new ``IRanges`` is returned with the
            reflected intervals. Otherwise, the current object is directly
            modified and a reference to it is returned.
        """

        if not isinstance(bounds, IRanges):
            raise TypeError("'bounds' must be an IRanges object.")

        if len(bounds) != len(self):
            raise ValueError("'bounds' does not contain the same number of intervals.")

        output = self._define_output(in_place)
        output._start = ((2 * bounds.start) + bounds.width) - output.end
        return output

    def restrict(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        keep_all_ranges: bool = False,
    ) -> "IRanges":
        """Restrict ranges to a given start and end positions.

        Args:
            start (int, optional): Start position. Defaults to None.
            end (int, optional): End position. Defaults to None.
            keep_all_ranges (bool, optional): Whether to keep intervals that do not overlap with start and end.
                Defaults to False.

        Returns:
            IRanges: A new ``IRanges`` is returned with the
            restricted intervals.
        """
        if start is None and end is None:
            warn("Both 'start' and 'end' are 'None'.")
            return self._define_output(False)

        if start is not None:
            new_starts = clip(self.start, start, None)
        else:
            new_starts = self.start

        if end is not None:
            new_ends = clip(self.end, None, end + 1)
        else:
            new_ends = self.end

        new_starts = new_starts
        new_widths = new_ends - new_starts

        if keep_all_ranges is True:
            new_widths = clip(new_widths, 0, None)
        else:
            _flt_idx = where(new_widths > -1)
            new_starts = new_starts[_flt_idx]
            new_widths = new_widths[_flt_idx]

        return IRanges(new_starts, new_widths)

    def overlap_indices(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> List[int]:
        """Find overlaps with the start and end positions.

        Args:
            start (int, optional): Start position. Defaults to None.
            end (int, optional): End position. Defaults to None.

        Returns:
            List[int]: List of indices that overlap with the given range.
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

        return overlaps

    ########################
    #### set operations ####
    ########################

    # set operations
    def union(self, other: "IRanges") -> "IRanges":
        """Find union of intervals with `other`.

        Args:
            other (GenomicRanges): `IRanges` object.

        Raises:
            TypeError: If ``other`` is not `IRanges`.

        Returns:
            IRanges: A new `IRanges` object with all ranges.
        """

        if not isinstance(other, IRanges):
            raise TypeError("'other' is not an IRanges object.")

        all_starts = combine_seqs(self.start, other.start)
        all_widths = combine_seqs(self.width, other.width)

        output = IRanges(all_starts, all_widths)
        output = output.reduce(min_gap_width=0, drop_empty_ranges=True)
        return output

    def setdiff(self, other: "IRanges") -> "IRanges":
        """Find set difference with `other`.

        Args:
            other (GenomicRanges): `IRanges` object.

        Raises:
            TypeError: If ``other`` is not `IRanges`.

        Returns:
            IRanges: A new `IRanges` object.
        """

        if not isinstance(other, IRanges):
            raise TypeError("'other' is not an IRanges object.")

        all_starts = combine_seqs(self.start, other.start)
        all_ends = combine_seqs(self.end, other.end)
        start = min(all_starts)
        end = max(all_ends)

        x_gaps = self.gaps(start=start, end=end)
        x_gaps_u = x_gaps.union(other)
        diff = x_gaps_u.gaps(start=start, end=end)

        return diff

    def intersect(self, other: "IRanges") -> "IRanges":
        """Find intersecting intervals with `other`.

        Args:
            other (GenomicRanges): `IRanges` object.

        Raises:
            TypeError: If ``other`` is not `IRanges`.

        Returns:
            IRanges: A new `IRanges` object with all intersecting intervals.
        """

        if not isinstance(other, IRanges):
            raise TypeError("'other' is not an IRanges object.")

        all_starts = combine_seqs(self.start, other.start)
        all_ends = combine_seqs(self.end, other.end)
        start = min(all_starts)
        end = max(all_ends)

        _gaps = other.gaps(start=start, end=end)
        _inter = self.setdiff(_gaps)

        return _inter


@combine_seqs.register
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
        start=combine_seqs(*[y._start for y in x]),
        width=combine_seqs(*[y._width for y in x]),
        names=all_names,
        mcols=combine_rows(*[y._mcols for y in x]),
        metadata=x[0]._metadata,
        validate=False,
    )
