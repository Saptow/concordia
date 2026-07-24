"""Bridge package for local education extensions."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_LOCAL_EDUCATION_PATH = Path(__file__).resolve().parents[2] / 'education'
_local_education_path_str = str(_LOCAL_EDUCATION_PATH)
if _local_education_path_str not in __path__:
  __path__.append(_local_education_path_str)
