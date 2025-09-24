"""Aplicación iqrl-bot."""

from __future__ import annotations

__all__ = [
    "load_config",
]

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """Carga la configuración YAML principal.

    Parameters
    ----------
    path:
        Ruta al archivo de configuración.

    Returns
    -------
    dict
        Diccionario con la configuración.
    """

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependencia opcional
        raise RuntimeError("PyYAML es requerido para cargar la configuración") from exc

    with open(Path(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
