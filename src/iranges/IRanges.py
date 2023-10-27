import biocutils as ut
from typing import Sequence, Optional, List, Union, Dict
from biocframe import BiocFrame
import numpy as np


class IRanges:
    def __init__(
        self, 
        start: Sequence[int], 
        width: Sequence[int], 
        names: Optional[Sequence[str]] = None, 
        mcols: Optional[BiocFrame] = None, 
        metadata: Optional[Dict] = None, 
        validate: bool = True,
    ):
        self._start = self._sanitize_start(start)
        self._width = self._sanitize_width(width)
        self._names = self._sanitize_names(names)
        self._mcols = self._sanitize_mcols(mcols)
        self._metadata = self._sanitize_metadata(metadata)

        if validate:
            self._validate_start()
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
        if self._names is None:
            return None
        elif not isinstance(names, list):
            names = list(names)
        return names

    def _validate_names(self):
        if self._names is None:
            return
        if ut.is_list_of_type(self._names, str):
            raise ValueError("'names' should be a list of strings")
        if len(self._names) != len(self._start):
            raise ValueError("'names' and 'start' should have the same length")

    def _sanitize_mcols(self, mcols):
        if mcols is None:
            return BiocFrame({}, number_of_rows=len(self._start))
        else:
            mcols

    def _validate_mcols(self):
        if not isinstance(self._mcols, BiocFrame)
            raise TypeError("'mcols' should be a BiocFrame")
        if self._mcols.shape[0] != len(self._start):
            raise ValueError("number of rows of 'mcols' should be equal to length of 'start'")

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
        return self._start

    def get_width(self) -> np.ndarray:
        return self._width

    def get_end(self) -> np.ndarray:
        return self._start + self._width

    def get_names(self) -> Union[None, List[str]]:
        return self._names

    def get_mcols(self) -> BiocFrame:
        return self._mcols

    def get_metadata(self) -> Dict:
        return self._metadata

    def _define_output(self, in_place):
        if in_place:
            return self
        else:
            return type(self)(self._start, self._width, names=self._names, mcols=self._mcols, metadata=self._metadata)

    def set_start(self, start: Sequence[int], in_place: bool = False) -> "IRanges":
        output = self._define_output(in_place)
        output._start = output._sanitize_start(start)
        output._validate_start()
        return output
        
    def set_width(self, width: Sequence[int], in_place: bool = False) -> "IRanges":
        output = self._define_output(in_place)
        output._width = output._sanitize_width(width)
        output._validate_width()
        return output

    def set_names(self, names: Optional[Sequence[str]], in_place: bool = False) -> "IRanges":
        output = self._define_output(in_place)
        output._names = output._sanitize_names(names)
        output._validate_names()
        return output

    def set_mcols(self, mcols: Optional[BiocFrame], in_place: bool = False) -> "IRanges":
        output = self._define_output(in_place)
        output._width = output._sanitize_mcols(mcols)
        output._validate_mcols()
        return output

    def set_metadata(self, metadata: Optional[Dict], in_place: bool = False) -> "IRanges":
        output = self._define_output(in_place)
        output._width = output._sanitize_metadata(metadata)
        output._validate_metadata()
        return output

    #########################
    #### Getitem/setitem ####
    #########################

    def _normalize_subscript(args: Union[slice, Sequence, int, str], length: int, names: Optional[List[str]]) -> Tuple:
        if isinstance(args, int):



    def __getitem__(self, args: Union[Sequence, int, str]) -> "IRanges":

        if isinstance(
        return





