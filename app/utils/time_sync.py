"""Sincronización de tiempo con tolerancia a fallos."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

try:
    import ntplib
except ImportError:  # pragma: no cover - dependencia opcional
    ntplib = None  # type: ignore

from .logger import get_logger


class TimeSynchronizer:
    """Gestor de sincronización horario basado en NTP.

    Si no es posible acceder a servidores NTP, utiliza el reloj local y expone
    métodos seguros para obtener timestamps y detectar el cierre de velas.
    """

    def __init__(self, server: str = "pool.ntp.org", refresh: int = 300) -> None:
        self._server = server
        self._refresh = refresh
        self._offset = 0.0
        self._lock = threading.Lock()
        self._logger = get_logger("time_sync")
        self._last_sync: float = 0.0

    def sync(self) -> None:
        """Sincroniza el reloj y actualiza el offset."""

        if ntplib is None:
            self._logger.warning("ntplib no disponible, usando reloj local")
            return
        try:
            client = ntplib.NTPClient()
            response = client.request(self._server, version=3)
            with self._lock:
                self._offset = response.offset
                self._last_sync = time.time()
            self._logger.info(
                "Sincronización NTP completada (offset=%.4fs)", self._offset
            )
        except Exception as exc:  # pragma: no cover - dependencias externas
            self._logger.error("Fallo al sincronizar NTP: %s", exc)

    def now(self) -> float:
        """Devuelve la hora actual sincronizada en segundos."""

        with self._lock:
            if time.time() - self._last_sync > self._refresh:
                threading.Thread(target=self.sync, daemon=True).start()
            return time.time() + self._offset

    def wait_until(self, timestamp: float) -> None:
        """Bloquea hasta el timestamp indicado (con chequeos periódicos)."""

        while True:
            now = self.now()
            if now >= timestamp:
                break
            time.sleep(min(0.25, max(0.01, timestamp - now)))

    def next_candle_close(self, timeframe: int) -> float:
        """Calcula el timestamp del próximo cierre de vela para un timeframe."""

        now = self.now()
        return now - (now % timeframe) + timeframe

    def datetime(self) -> datetime:
        """Devuelve un objeto datetime timezone-aware."""

        return datetime.fromtimestamp(self.now(), tz=timezone.utc)
