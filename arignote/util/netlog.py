"""
Logging setup for the Arignote library.
"""
__author__ = 'shoover'

import logging
import io
import sys


debug_logging = io.StringIO()  # Global bucket for debug statements.
DEBUG_FORMAT = "%(asctime)s %(levelname)8s [%(process)6d %(module)12s.%(funcName)-12s:%(lineno)4d] %(message)s"
SCREEN_FORMAT = "%(asctime)s | %(message)s"
if sys.version_info.major == 2:
    DEBUG_FORMAT = unicode(DEBUG_FORMAT)


def setup_logging(name="nnets", level="INFO"):
    log = logging.getLogger(name=name)
    log.handlers = []

    # Log all messages to a global string stream which we can write into our checkpoints.
    handler_debug = logging.StreamHandler(debug_logging)
    handler_debug.setFormatter(logging.Formatter(DEBUG_FORMAT))
    handler_debug.setLevel("DEBUG")
    log.addHandler(handler_debug)

    # Allow modules to have their own debug log, if they want.
    local_debug_logging = io.StringIO()
    local_debug = logging.StreamHandler(local_debug_logging)
    local_debug.setFormatter(logging.Formatter(DEBUG_FORMAT))
    local_debug.setLevel("DEBUG")
    log.addHandler(local_debug)

    # Print "INFO" and above messages to the screen.
    handler_screen = logging.StreamHandler(sys.stdout)
    handler_screen.setFormatter(logging.Formatter(DEBUG_FORMAT if level=="DEBUG" else SCREEN_FORMAT,
                                                  datefmt="%H:%M:%S"))
    handler_screen.setLevel(level)
    log.addHandler(handler_screen)

    log.setLevel("DEBUG")  # Let the logger catch all emits. The handlers have their own levels.

    log.debug_global = debug_logging

    return log


default_log = setup_logging(name="nnets", level="INFO")
