from typing import Optional, Tuple, Union

import numpy as np

from .utils import handle_negative_coords, normalize_array

__author__ = "Jayaram Kancherla"
__copyright__ = "jkanche"
__license__ = "MIT"

# reference: https://github.com/Bioconductor/IRanges/blob/devel/R/IRanges-constructor.R#L201


class SEWWrangler:
    """Handler to resolve start/end/width parameters."""

    def __init__(
        self,
        ref_widths: np.ndarray,
        start: Optional[Union[int, np.ndarray]] = None,
        end: Optional[Union[int, np.ndarray]] = None,
        width: Optional[Union[int, np.ndarray]] = None,
        translate_negative: bool = True,
        allow_nonnarrowing: bool = False,
    ):
        """Initialize SEW parameters.

        Args:
            ref_widths:
                Reference widths array.

            start:
                Start positions.

            end:
                End positions.

            width:
                Widths.

            translate_negative:
                Whether to translate negative coordinates.

            allow_nonnarrowing:
                Whether to allow ranges wider than reference.
        """
        self.ref_widths = np.asarray(ref_widths, dtype=np.int32)
        self.length = len(ref_widths)
        self.allow_nonnarrowing = allow_nonnarrowing

        self.start = normalize_array(start, self.length)
        self.end = normalize_array(end, self.length)
        self.width = normalize_array(width, self.length)

        if translate_negative:
            self.start = handle_negative_coords(self.start, self.ref_widths)
            self.end = handle_negative_coords(self.end, self.ref_widths)

        if not allow_nonnarrowing:
            # validate supplied ends
            if not self.end.mask.all():
                too_wide = (~self.end.mask) & (self.end > self.ref_widths)
                if np.any(too_wide):
                    idx = np.where(too_wide)[0][0]
                    raise Exception(
                        f"solving row {idx + 1}: 'allow.nonnarrowing' is FALSE and "
                        f"the supplied end ({int(self.end[idx])}) is > refwidth"
                    )

    def _validate_narrowing(self, starts: np.ndarray, widths: np.ndarray) -> None:
        """Validate that ranges don't exceed reference width."""
        if not self.allow_nonnarrowing:
            ends = starts + widths - 1
            too_wide = ends > self.ref_widths
            if np.any(too_wide):
                idx = np.where(too_wide)[0][0]
                raise Exception(
                    f"solving row {idx + 1}: 'allow.nonnarrowing' is FALSE and "
                    f"the solved end ({int(ends[idx])}) is > refwidth"
                )

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve Start/End/Width parameters to concrete ranges.

        Returns:
            Tuple of resolved (starts, widths) ranges.
        """
        out_starts = np.ones(self.length, dtype=np.int32)
        out_widths = self.ref_widths.copy()

        if not self.width.mask.all():
            if np.any((~self.width.mask) & (self.width < 0)):
                raise Exception("negative values are not allowed in 'width'")

            if not self.end.mask.all():
                # Width and end specified
                # mask = (~self.width.mask) & (~self.end.mask)
                out_starts = self.end - self.width + 1
                out_widths = self.width
                # Validate after computing
                self._validate_narrowing(out_starts, out_widths)

            elif not self.start.mask.all():
                # Width and start specified
                # mask = (~self.width.mask) & (~self.start.mask)
                out_starts = self.start
                out_widths = self.width
                # Validate after computing
                self._validate_narrowing(out_starts, out_widths)

            else:
                # Only width specified
                out_widths = self.width

        # Handle start/end specification
        elif not self.start.mask.all() and not self.end.mask.all():
            out_starts = self.start
            out_widths = self.end - self.start + 1
            if np.any(out_widths < 0):
                raise Exception("ranges contain negative width")
            # Validate after computing
            self._validate_narrowing(out_starts, out_widths)

        # Handle only start
        elif not self.start.mask.all():
            out_starts = self.start
            out_widths = self.ref_widths - (self.start - 1)
            # Validate after computing
            self._validate_narrowing(out_starts, out_widths)

        # Handle only end
        elif not self.end.mask.all():
            out_widths = self.end

        # Validate after computing
        self._validate_narrowing(out_starts, out_widths)
        return out_starts, out_widths
