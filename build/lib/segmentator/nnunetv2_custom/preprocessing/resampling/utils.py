from typing import Callable

from batchgenerators.utilities.file_and_folder_operations import join

import segmentator.nnunetv2_custom
from segmentator.nnunetv2_custom.utilities.find_class_by_name import (
    recursive_find_python_class,
)


def recursive_find_resampling_fn_by_name(resampling_fn: str) -> Callable:
    ret = recursive_find_python_class(
        join(segmentator.nnunetv2_custom.__path__[0], "preprocessing", "resampling"),
        resampling_fn,
        "segmentator.nnunetv2_custom.preprocessing.resampling",
    )
    if ret is None:
        raise RuntimeError(
            "Unable to find resampling function named '%s'. Please make sure this fn is located in the "
            "segmentator.nnunetv2_custom.preprocessing.resampling module."
            % resampling_fn
        )
    else:
        return ret
