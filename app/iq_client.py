"""Cliente resiliente para IQ Option."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from .utils.logger import get_logger

try:  # pragma: no cover - import pesado
    from iqoptionapi.stable_api import IQ_Option
except Exception:  # pragma: no cover - entorno de test sin dependencia
    IQ_Option = None  # type: ignore


@dataclass(slots=True)
class AssetOpenTime:
    name: str
    is_open: bool
    option_type: str


class IQOptionClient:
    """Wrapper con reconexión automática sobre iqoptionapi."""

    def __init__(
        self,
        email: str,
        password: str,
        account_type: str = "PRACTICE",
        reconnect_interval: int = 5,
    ) -> None:
        load_dotenv()
        self.email = email
        self.password = password
        self.account_type = account_type.upper()
        self.reconnect_interval = reconnect_interval
        self._client: Optional[IQ_Option] = None
        self._lock = threading.RLock()
        self._logger = get_logger("iq_client")
        self._connect()

    def _connect(self) -> None:
        if IQ_Option is None:
            raise RuntimeError("iqoptionapi no disponible en el entorno actual")
        self._logger.info("Conectando con IQ Option como %s", self.email)
        self._client = IQ_Option(self.email, self.password)
        self._client.connect()
        if not self._client.check_connect():
            raise ConnectionError("No se pudo establecer conexión con IQ Option")
        self._client.change_balance(self.account_type)
        self._logger.info("Conectado. Balance %s", self.account_type)

    def ensure_connection(self) -> None:
        with self._lock:
            if self._client is None:
                self._connect()
            elif not self._client.check_connect():
                self._logger.warning("Reconectando con IQ Option")
                time.sleep(self.reconnect_interval)
                self._connect()

    @property
    def client(self) -> IQ_Option:
        if self._client is None:
            raise RuntimeError("Cliente no inicializado")
        return self._client

    def get_open_assets(self) -> List[AssetOpenTime]:
        self.ensure_connection()
        result = self.client.get_all_open_time()
        assets: List[AssetOpenTime] = []
        for market in ("turbo", "binary", "digital"):
            for asset, info in result.get(market, {}).items():
                assets.append(
                    AssetOpenTime(
                        name=asset,
                        is_open=bool(info.get("open", False)),
                        option_type=market,
                    )
                )
        return assets

    def start_candles_stream(self, asset: str, timeframe: int, count: int) -> None:
        self.ensure_connection()
        self.client.start_candles_stream(asset, timeframe, count)

    def stop_candles_stream(self, asset: str, timeframe: int) -> None:
        self.ensure_connection()
        self.client.stop_candles_stream(asset, timeframe)

    def get_realtime_candles(self, asset: str, timeframe: int) -> Dict[int, Dict[str, float]]:
        self.ensure_connection()
        return self.client.get_realtime_candles(asset, timeframe)

    def buy(self, asset: str, amount: float, direction: str, duration: int) -> Tuple[bool, Optional[int]]:
        self.ensure_connection()
        success, trade_id = self.client.buy(amount, asset, direction, duration)
        return bool(success), trade_id

    def buy_digital(self, asset: str, amount: float, direction: str, duration: int) -> Tuple[bool, Optional[int]]:
        self.ensure_connection()
        trade_id = self.client.buy_digital_spot(asset, amount, direction, duration)
        return trade_id is not None, trade_id

    def check_win(self, trade_id: int) -> Optional[float]:
        self.ensure_connection()
        status, profit = self.client.check_win_v2(trade_id)
        if status == "win":
            return float(profit)
        if status == "loose":
            return -float(abs(profit))
        if status == "equal":
            return 0.0
        return None

    def change_balance(self, account_type: str) -> None:
        self.account_type = account_type.upper()
        self.ensure_connection()
        self.client.change_balance(self.account_type)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
            self._client = None
