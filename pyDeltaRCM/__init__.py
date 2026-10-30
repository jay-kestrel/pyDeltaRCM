import logging

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM.preprocessor import Preprocessor
from pyDeltaRCM.shared_tools import _get_version

__all__ = ['DeltaModel', 'Preprocessor']
__version__: str = _get_version()

# library-safety contract: the package never configures the root logger and
# never attaches real handlers at import time. The NullHandler stops records
# logged on the package logger from falling through to the last-resort
# handler when a host application has not configured logging.
logging.getLogger("pyDeltaRCM").addHandler(logging.NullHandler())
