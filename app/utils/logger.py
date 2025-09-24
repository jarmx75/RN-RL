"""Configuración de logging estructurado."""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "iqrl") -> logging.Logger:
    """Obtiene un logger configurado con formato estructurado.

    Parameters
    ----------
    name:
        Nombre del logger.

    Returns
    -------
    logging.Logger
        Instancia de logger con configuración estándar para el proyecto.
    """

    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER.getChild(name)

    log_level = os.getenv("IQRL_LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("iqrl")
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = Path(os.getenv("IQRL_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "iqrl.log", maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    _LOGGER = logger
    return logger.getChild(name)
