import threading
import yaml
from app.exec.runner_live import main as runner_live_main

class Controller:
    def __init__(self, gui):
        with open("config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        self.gui = gui
        self.assets = cfg["assets"]
        self.timeframes = cfg["timeframes"]
        self.threshold = cfg["threshold"]
        self.amount = cfg["amount"]
        self.thread = None

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run_live, daemon=True)
            self.thread.start()
            self.gui.update_status("Corriendo")

    def stop(self):
        if self.thread and self.thread.is_alive():
            # Not actual signal for runner_live, but placeholder for demonstration
            self.gui.update_status("Detenido")

    def run_live(self):
        runner_live_main()