"""Interfaz PyQt6 para iqrl-bot."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import List

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from .controller import Controller
from .utils.logger import get_logger


class MetricLabel(QLabel):
    def update_value(self, value: float, suffix: str = "") -> None:
        self.setText(f"{value:.2f}{suffix}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger("gui")
        self.controller = Controller()
        self.setWindowTitle("iqrl-bot")
        self.resize(1400, 900)
        self._build_ui()
        self._connect_signals()
        self._update_status(False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_clock)
        self.timer.start(1000)

    # UI
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        header = self._build_header()
        layout.addLayout(header)

        controls = self._build_controls()
        layout.addWidget(controls)

        metrics = self._build_metrics()
        layout.addWidget(metrics)

        tables = self._build_tables()
        layout.addLayout(tables)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        layout.addWidget(self.log_console)

        self.setCentralWidget(central)

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.connection_label = QLabel("Desconectado")
        self.connection_label.setAutoFillBackground(True)
        self.balance_label = QLabel("Balance: 0.00")
        self.mode_label = QLabel("Modo: DEMO")
        self.clock_label = QLabel("--:--:--")
        for widget in [self.connection_label, self.balance_label, self.mode_label, self.clock_label]:
            layout.addWidget(widget)
        layout.addStretch(1)
        return layout

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Control")
        layout = QGridLayout()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.download_button = QPushButton("Descargar históricos")
        self.train_button = QPushButton("Entrenar modelo")
        self.backtest_button = QPushButton("Backtest")
        self.demo_button = QPushButton("Ejecutar Demo")
        self.real_button = QPushButton("Ejecutar Real")

        layout.addWidget(self.start_button, 0, 0)
        layout.addWidget(self.stop_button, 0, 1)
        layout.addWidget(self.download_button, 0, 2)
        layout.addWidget(self.train_button, 0, 3)
        layout.addWidget(self.backtest_button, 0, 4)
        layout.addWidget(self.demo_button, 1, 0)
        layout.addWidget(self.real_button, 1, 1)

        assets_box = QGroupBox("Activos")
        assets_layout = QVBoxLayout()
        self.asset_checks: List[QCheckBox] = []
        for asset in self.controller.state.assets:
            cb = QCheckBox(asset)
            cb.setChecked(True)
            self.asset_checks.append(cb)
            assets_layout.addWidget(cb)
        assets_box.setLayout(assets_layout)
        layout.addWidget(assets_box, 2, 0, 1, 2)

        tf_box = QGroupBox("Timeframes")
        tf_layout = QVBoxLayout()
        self.tf_checks: List[QCheckBox] = []
        for tf in self.controller.state.timeframes:
            cb = QCheckBox(f"{tf//60}m")
            cb.setChecked(True)
            cb.tf_value = tf  # type: ignore[attr-defined]
            self.tf_checks.append(cb)
            tf_layout.addWidget(cb)
        tf_box.setLayout(tf_layout)
        layout.addWidget(tf_box, 2, 2, 1, 1)

        risk_box = QGroupBox("Riesgo")
        risk_layout = QFormLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(95)
        self.threshold_slider.setValue(int(self.controller.state.threshold * 100))
        risk_layout.addRow("Umbral (%)", self.threshold_slider)

        self.amount_spin = QSpinBox()
        self.amount_spin.setMaximum(1000)
        self.amount_spin.setValue(int(self.controller.state.amount))
        risk_layout.addRow("Monto fijo", self.amount_spin)

        self.amount_percent = QLineEdit("0.02")
        risk_layout.addRow("Monto %", self.amount_percent)

        self.dynamic_threshold_toggle = QCheckBox("Umbral dinámico")
        risk_layout.addRow(self.dynamic_threshold_toggle)

        risk_box.setLayout(risk_layout)
        layout.addWidget(risk_box, 2, 3, 1, 2)

        box.setLayout(layout)
        return box

    def _build_metrics(self) -> QWidget:
        box = QGroupBox("Métricas")
        layout = QHBoxLayout()
        self.win_rate = MetricLabel("0.00%")
        self.ops_hour = MetricLabel("0.0")
        self.profit_factor = MetricLabel("0.0")
        self.drawdown = MetricLabel("0.0")
        self.pl = MetricLabel("0.0")
        for label, title in [
            (self.win_rate, "Win-rate"),
            (self.ops_hour, "Ops/h"),
            (self.profit_factor, "PF"),
            (self.drawdown, "DD"),
            (self.pl, "P/L"),
        ]:
            group = QVBoxLayout()
            group.addWidget(QLabel(title))
            group.addWidget(label)
            layout.addLayout(group)
        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setBackground("w")
        self.equity_curve = self.equity_plot.plot([], [], pen=pg.mkPen(color=(0, 128, 255), width=2))
        layout.addWidget(self.equity_plot)
        box.setLayout(layout)
        return box

    def _build_tables(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Hora", "Activo", "TF", "Dirección", "Prob", "Monto", "Payout", "Resultado"]
        )
        layout.addWidget(self.table)
        return layout

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(lambda: self.controller.run_live("PRACTICE"))
        self.stop_button.clicked.connect(self._stop_bot)
        self.download_button.clicked.connect(self._download_history)
        self.train_button.clicked.connect(self.controller.start_training)
        self.backtest_button.clicked.connect(lambda: self.controller.run_backtest("artifacts/models/ppo_latest.zip"))
        self.demo_button.clicked.connect(lambda: self.controller.run_live("PRACTICE"))
        self.real_button.clicked.connect(lambda: self.controller.run_live("REAL"))
        self.controller.signals.log.connect(self._append_log)

    def _stop_bot(self) -> None:
        QMessageBox.information(self, "Stop", "Para detener el bot cierra la ventana live")

    def _download_history(self) -> None:
        self._append_log("Descarga de datos no implementada en modo offline")

    def _append_log(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {message}")

    def _update_status(self, connected: bool) -> None:
        palette = self.connection_label.palette()
        color = QColor("green" if connected else "red")
        palette.setColor(QPalette.ColorRole.WindowText, color)
        self.connection_label.setPalette(palette)
        self.connection_label.setText("Conectado" if connected else "Desconectado")

    def _update_clock(self) -> None:
        self.clock_label.setText(datetime.utcnow().strftime("%H:%M:%S"))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
