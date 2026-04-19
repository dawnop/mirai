from .log import logger as _logger  # noqa: F401  — must precede torch imports
from .decorator import op, module
from .build import build

__version__ = "0.1.0"
