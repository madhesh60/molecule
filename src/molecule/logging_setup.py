"""logging_setup.py — file logger for the MCP server.

Logs go to .molecule/molecule.log (never stdout — stdout is the MCP
stdio JSON-RPC channel and must stay clean). Falls back to stderr when
log_to_file is disabled or the file can't be opened.
"""
from __future__ import annotations

import logging
import sys

from molecule.config import MoleculeConfig

_LOGGER_NAME = "molecule"


def get_logger(config: MoleculeConfig) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)

    handler: logging.Handler
    if config.log_to_file:
        try:
            log_path = config.db_path.parent / "molecule.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
        except OSError:
            handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    return logger
