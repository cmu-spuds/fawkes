# -*- coding: utf-8 -*-
# @Date    : 2020-07-01
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/


__version__ = "1.1.1"

from .differentiator import FawkesMaskGeneration
from .protection import main, Fawkes
from .utils import (
    load_extractor,
    dump_image,
    reverse_process_cloaked,
    Faces,
    filter_image_paths,
)

__all__ = (
    "__version__",
    "FawkesMaskGeneration",
    "load_extractor",
    "dump_image",
    "reverse_process_cloaked",
    "Faces",
    "filter_image_paths",
    "main",
    "Fawkes",
)
