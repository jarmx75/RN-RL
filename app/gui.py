import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QLabel, QCheckBox, QSlider, QComboBox, QSpinBox
)
from app.controller import Controller

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("iqrl-bot")
        self.controller = Controller(self)
        layout = QVBoxLayout()

        self.status = QLabel("Desconectado")
        layout.addWidget(self.status)

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.asset_checks = []
        for asset in self.controller.assets:
            cb = QCheckBox(asset)
            cb.setChecked(True)
            self.asset_checks.append(cb)
            layout.addWidget(cb)

        self.tf_combo = QComboBox()
        for tf in self.controller.timeframes:
            self.tf_combo.addItem(f"{tf//60}m", tf)
        layout.addWidget(QLabel("Timeframe"))
        layout.addWidget(self.tf_combo)

        self.threshold_slider = QSlider()
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(90)
        self.threshold_slider.setValue(int(self.controller.threshold * 100))
        layout.addWidget(QLabel("Umbral (%)"))
        layout.addWidget(self.threshold_slider)

        self.amount_spin = QSpinBox()
        self.amount_spin.setMaximum(100)
        self.amount_spin.setValue(int(self.controller.amount))
        layout.addWidget(QLabel("Monto"))
        layout.addWidget(self.amount_spin)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Hora","Activo","TF","Dirección","Prob","Monto","Payout","Resultado"])
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.start_btn.clicked.connect(self.controller.start)
        self.stop_btn.clicked.connect(self.controller.stop)

    def update_status(self, text: str):
        self.status.setText(text)

    def add_trade(self, data: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, key in enumerate(["time","asset","tf","direction","prob","amount","payout","resultado"]):
            item = QTableWidgetItem(str(data.get(key, "")))
            self.table.setItem(row, col, item)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()