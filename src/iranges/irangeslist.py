from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import biocutils as ut
from compressed_lists import CompressedList, Partitioning
from compressed_lists.split_generic import _generic_register_helper, splitAsCompressedList

from .IRanges import IRanges, _combine_IRanges

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CompressedIRangesList(CompressedList):
    """CompressedList for IRanges."""

    def __init__(
        self,
        unlist_data: IRanges,
        partitioning: Partitioning,
        element_metadata: Optional[dict] = None,
        metadata: Optional[Union[Dict[str, Any], ut.NamedList]] = None,
        **kwargs,
    ):
        """Initialize a CompressedIRangesList.

        Args:
            unlist_data:
                IRanges object.

            partitioning:
                Partitioning object defining element boundaries.

            element_metadata:
                Optional metadata for elements.

            metadata:
                Optional general metadata.

            kwargs:
                Additional arguments.
        """
        if not isinstance(unlist_data, IRanges):
            raise TypeError("'unlist_data' is not a `IRanges` object.")

        super().__init__(
            unlist_data, partitioning, element_type=IRanges, element_metadata=element_metadata, metadata=metadata
        )

    @classmethod
    def from_list(
        cls,
        lst: List[IRanges],
        names: Optional[Union[ut.Names, Sequence[str]]] = None,
        metadata: Optional[Union[Dict[str, Any], ut.NamedList]] = None,
    ) -> CompressedIRangesList:
        """Create a `CompressedIRangesList` from a regular list.

        This concatenates the list of `IRanges` objects.

        Args:
            lst:
                List of `IRanges` objects.

                Must have the same number and names of columns.

            names:
                Optional names for list elements.

            metadata:
                Optional metadata.

        Returns:
            A new `CompressedList`.
        """
        unlist_data = _combine_IRanges(*lst)
        partitioning = Partitioning.from_list(lst, names)
        return cls(unlist_data, partitioning, metadata=metadata)

    def extract_range(self, start: int, end: int) -> IRanges:
        """Extract a range from `unlist_data`.

        This method must be implemented by subclasses to handle
        type-specific extraction from `unlist_data`.

        Args:
            start:
                Start index (inclusive).

            end:
                End index (exclusive).

        Returns:
            Extracted element.
        """
        try:
            return ut.subset_sequence(self._unlist_data, range(start, end))
        except Exception as e:
            raise NotImplementedError(
                "Custom classes should implement their own `extract_range` method for slice operations"
            ) from e

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}(number_of_elements={len(self)}"
        output += ", unlist_data=" + self._unlist_data.__repr__()
        output += ", partitioning=" + self._partitioning.__repr__()
        output += (
            ", element_type=" + self._element_type.__name__
            if not isinstance(self._element_type, str)
            else self._element_type
        )

        output += ", element_metadata=" + self._element_metadata.__repr__()

        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)

        output += ")"
        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"

        output += f"number of elements: ({len(self)}) of type: {self._element_type.__name__ if not isinstance(self._element_type, str) else self._element_type}\n"

        output += f"unlist_data: {self._unlist_data.__str__()}\n"

        output += f"partitioning: {ut.print_truncated_list(self._partitioning)}\n"

        output += f"element_metadata({str(len(self._element_metadata))} rows): {ut.print_truncated_list(list(self._element_metadata.get_column_names()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"
        output += f"metadata({str(len(self._metadata))}): {ut.print_truncated_list(list(self._metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output


@splitAsCompressedList.register
def _(
    data: IRanges,
    groups_or_partitions: Union[list, Partitioning],
    names: Optional[Union[ut.Names, Sequence[str]]] = None,
    metadata: Optional[dict] = None,
) -> CompressedIRangesList:
    """Handle lists of IRanges objects."""

    partitioned_data, groups_or_partitions = _generic_register_helper(
        data=data, groups_or_partitions=groups_or_partitions, names=names
    )

    if not isinstance(partitioned_data, IRanges) and len(partitioned_data) != 0:
        partitioned_data = _combine_IRanges(*partitioned_data)

    return CompressedIRangesList(unlist_data=partitioned_data, partitioning=groups_or_partitions, metadata=metadata)
