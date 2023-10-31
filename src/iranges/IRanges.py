import biocutils as ut
from typing import Sequence, Optional, List, Union, Dict
from biocframe import BiocFrame
from biocgenerics import combine_seqs, combine_rows, show_as_cell, format_table
import numpy as np
from copy import deepcopy


class IRanges:
    """A collection of integer ranges, equivalent to the ``IRanges`` class from the Bioconductor package of the same
    name.

    This holds a start position and a width, and is most typically used to represent coordinates along some genomic
    sequence. The interpretation of the start position depends on the application; for sequences, the start is usually a
    1-based position, but other use cases may allow zero or even negative values.
    """

    def __init__(
        self,
        start: Sequence[int],
        width: Sequence[int],
        names: Optional[Sequence[str]] = None,
        mcols: Optional[BiocFrame] = None,
        metadata: Optional[Dict] = None,
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
        if isinstance(start, np.ndarray) and start.dtype == np.int32:
            return start
        return np.array(start, dtype=np.int32, copy=False)

    def _sanitize_width(self, width):
        if isinstance(width, np.ndarray) and width.dtype == np.int32:
            return width
        return np.array(width, dtype=np.int32, copy=False)

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
            return
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
                "number of rows of 'mcols' should be equal to length of 'start'"
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

    def get_start(self) -> np.ndarray:
        """
        Returns:
            NumPy array of 32-bit signed integers containing the start
            positions for all ranges.
        """
        return self._start

    def get_width(self) -> np.ndarray:
        """
        Returns:
            NumPy array of 32-bit signed integers containing the widths for all
            ranges.
        """
        return self._width

    def get_end(self) -> np.ndarray:
        """
        Returns:
            NumPy array of 32-bit signed integers containing the end position
            (not inclusive) for all ranges.
        """
        return self._start + self._width

    def get_names(self) -> Union[None, List[str]]:
        """
        Returns:
            List containing the names for all ranges, or None if no names are
            present.
        """
        return self._names

    def get_mcols(self) -> BiocFrame:
        """
        Returns:
            Data frame containing additional metadata columns for all ranges.
        """
        return self._mcols

    def get_metadata(self) -> Dict:
        """
        Returns:
            Dictionary containing additional metadata.
        """
        return self._metadata

    def _define_output(self, in_place):
        if in_place:
            return self
        else:
            return self.__copy__()

    def set_start(self, start: Sequence[int], in_place: bool = False) -> "IRanges":
        """
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

    def set_mcols(
        self, mcols: Optional[BiocFrame], in_place: bool = False
    ) -> "IRanges":
        """
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

    def set_metadata(
        self, metadata: Optional[Dict], in_place: bool = False
    ) -> "IRanges":
        """
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
        """
        Args:
            subset:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

        Returns:
            A new ``IRanges`` object containing the ranges of interest.
        """
        idx, scalar = ut.normalize_subscript(subset, len(self), self._names)
        return type(self)(
            start=self._start[idx],
            width=self._width[idx],
            names=ut.subset(self._names, idx) if self._names is not None else None,
            mcols=self._mcols[list(idx), :],  # doesn't support ranges yet.
            metadata=self._metadata,
        )

    def __setitem__(
        self, args: Union[Sequence, int, str, bool, slice, range], value: "IRanges"
    ):
        """
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
        idx, scalar = ut.normalize_subscript(args, len(self), self._names)
        self._start[idx] = value._start
        self._width[idx] = value._width
        #        self._mcols[list(idx),:] = value._mcols # doesn't support ranges yet.

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
        """
        Returns:
            A string representation of this object.
        """
        opt = np.get_printoptions()
        old_threshold = opt["threshold"]
        old_edgeitems = opt["edgeitems"]

        try:
            opt["threshold"] = 50
            opt["edgeitems"] = 3
            message = "IRanges(start=" + repr(self._start)
            message += ", width=" + repr(self._width)
            if self._names:
                message += ", names=" + ut.print_truncated_list(self._names)
            if self._mcols.shape[1] > 0:
                message += ", mcols=" + repr(self._mcols)
            if len(self._metadata):
                message += ", metadata=" + repr(self._metadata)
            message += ")"
        finally:
            opt["threshold"] = old_threshold
            opt["edgeitems"] = old_edgeitems

        return message

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string representation of this object.
        """
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

            if self._names is not None:
                raw_floating = ut.subset(self._names, indices)
            else:
                raw_floating = ["[" + str(i) + "]" for i in indices]
            if insert_ellipsis:
                raw_floating = raw_floating[:3] + [""] + raw_floating[3:]
            floating = ["", ""] + raw_floating

            columns = []
            for prop in [ "start", "end", "width" ]:
                data = getattr(self, "get_" + prop)()
                showed = show_as_cell(data, indices)
                header = [prop, "<" + type(data).__name__ + ">"]
                if insert_ellipsis:
                    showed = showed[:3] + ["..."] + showed[3:]
                columns.append(header + showed)

            if self._mcols.shape[1] > 0:
                spacer = ["|"] * (len(indices) + insert_ellipsis)
                columns.append(["", ""] + spacer)

                for col in self._mcols.get_column_names():
                    data = self._mcols.column(col)
                    showed = show_as_cell(data, indices)
                    header = [col, "<" + type(data).__name__ + ">"]
                    minwidth = max(40, len(header[0]), len(header[1]))
                    for i, y in enumerate(showed):
                        if len(y) > minwidth:
                            showed[i] = y[: minwidth - 3] + "..."
                    if insert_ellipsis:
                        showed = showed[:3] + ["..."] + showed[3:]
                    columns.append(header + showed)

            output += format_table(columns, floating_names=floating)
            added_table = True

        footer = []
        if len(self._metadata):
            footer.append(
                "metadata ("
                + str(len(self._metadata))
                + "): "
                + ut.print_truncated_list(
                    list(self._metadata.keys()), sep=" ", include_brackets=False
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

    def __copy__(self):
        """
        Returns:
            A shallow copy of this object.
        """
        return type(self)(
            start=self._start,
            width=self._width,
            names=self._names,
            mcols=self._mcols,
            metadata=self._metadata,
            validate=False,
        )

    def __deepcopy__(self, memo):
        """
        Args:
            memo: Passed to internal :py:meth:`~deepcopy` calls.

        Returns:
            A deep copy of this object.
        """
        return type(self)(
            start=deepcopy(self._start, memo),
            width=deepcopy(self._width, memo),
            names=deepcopy(self._names, memo),
            mcols=deepcopy(self._mcols, memo),
            metadata=deepcopy(self._metadata, memo),
            validate=False,
        )


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
