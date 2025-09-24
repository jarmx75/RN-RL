# iqrl-bot

iqrl-bot es una plataforma integral para trading algorítmico con IQ Option que integra recopilación de datos, entrenamiento de agentes de Aprendizaje por Refuerzo Profundo, backtesting y ejecución en vivo con una interfaz gráfica PyQt6. El proyecto se ha diseñado para funcionar en macOS (incluyendo MacBook Air M2) aprovechando PyTorch con soporte MPS cuando está disponible.

## Características clave

- **Conectividad IQ Option** mediante `iqoptionapi` con reconexión automática, sincronización horaria y soporte para cuentas práctica/real.
- **Datos multi-activo y multi-timeframe**: lectura en vivo de velas para 10+ pares y descargas históricas masivas con almacenamiento en CSV/SQLite.
- **Agente DRL** basado en PPO/A2C de Stable-Baselines3 con extractor CNN-LSTM capaz de fusionar información multi-temporal.
- **Gestión de riesgo profesional** configurable: límites diarios, pérdidas consecutivas, horarios operables, tamaño de posición fijo o porcentual y umbral dinámico.
- **Ejecución automática** sincronizada al cambio de vela para binarias/turbo y digitales con control de concurrencia por activo/TF.
- **Logging y métricas** detalladas con panel PyQt6 de una sola ventana que controla todo el flujo (descarga, entrenamiento, backtest, live demo/real).

## Quickstart

### 1. Preparación del entorno

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` con tus credenciales de IQ Option y selecciona el modo de operación (`PRACTICE` o `REAL`).

### 2. Descarga de datos históricos

```bash
bash scripts/download_data.sh
```

Este script descarga velas para los activos y timeframes definidos en `config/config.yaml` y las almacena en `data/` y en SQLite (`storage/trades.db`).

### 3. Entrenamiento del agente PPO

```bash
bash scripts/train.sh
```

El script lanza `app.exec.runner_train` que descarga los datos necesarios, crea entornos vectorizados (8 workers por defecto) y entrena un modelo PPO guardando checkpoints en `artifacts/models/` y registros de TensorBoard en `artifacts/tensorboard/`.

Para habilitar TensorBoard manualmente:

```bash
tensorboard --logdir artifacts/tensorboard
```

### 4. Backtest

```bash
bash scripts/backtest.sh
```

Genera métricas agregadas (win-rate, profit factor, drawdown máximo y P/L) y exporta los resultados a `artifacts/reports/backtest.csv`.

### 5. Ejecución en vivo

- **Demo** (cuenta práctica):

  ```bash
  bash scripts/live_demo.sh
  ```

- **Real** (cuenta real):

  ```bash
  bash scripts/live_real.sh
  ```

Ambos scripts arrancan la GUI `PyQt6` (`app.gui`) desde donde podrás iniciar/parar el bot, seleccionar activos/timeframes, ajustar umbrales, lanzar descargas históricas, entrenar, hacer backtest y ejecutar en demo/real.

## Uso de la GUI

La interfaz principal contiene:

- **Header** con estado de conexión (verde/rojo), reloj sincronizado, balance y modo (Demo/Real).
- **Controles** para Start/Stop, selección de activos (checkboxes), timeframes (1m/5m/15m), umbral (slider), monto fijo o % de balance, límites de riesgo (pérdidas consecutivas, stop diario, meta diaria, horarios permitidos), y un toggle para el umbral dinámico.
- **Botones de acción**: descargar históricos, entrenar modelo, backtest, ejecutar en demo, ejecutar en real.
- **Panel de métricas** con win-rate, operaciones/hora, profit factor, drawdown y P/L, además de un gráfico de equity en tiempo real.
- **Tabla** de operaciones con hora, activo, timeframe, dirección, probabilidad estimada, monto, payout y resultado.
- **Consola de eventos** para logs.

Las preferencias se guardan automáticamente en `config/config.yaml` al cerrar la aplicación.

## Notas de despliegue en Mac M2

- Utiliza Python 3.11+ instalado vía `pyenv` o `brew`.
- Instala `libomp` para PyTorch si es necesario (`brew install libomp`).
- PyTorch con soporte MPS se instala automáticamente cuando está disponible (`pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu`). Ajusta según tus necesidades.
- Asegúrate de conceder permisos de red a Python para permitir la conexión WebSocket con IQ Option.

## Estructura del repositorio

Consulta la sección "Árbol del repositorio" en la entrega final para ver la estructura completa.

## Scripts incluidos

- `scripts/setup_venv.sh`: crea/actualiza el entorno virtual `.venv` con Python 3.11.
- `scripts/download_data.sh`: descarga datos históricos.
- `scripts/train.sh`: entrena el agente PPO.
- `scripts/backtest.sh`: ejecuta backtesting sobre datos recientes.
- `scripts/live_demo.sh`: levanta la GUI en modo demo.
- `scripts/live_real.sh`: levanta la GUI en modo real.

## Testing

Ejecuta los tests (sanity + entorno) con:

```bash
pytest
```

Los tests utilizan datos sintéticos y un stub de cliente IQ Option para validar el pipeline offline.

## Seguridad

- Las credenciales se cargan desde `.env` y nunca se registran en texto plano.
- El bot aplica límites de riesgo estrictos antes de ejecutar cualquier operación.

## Licencia

Este proyecto se distribuye bajo la licencia MIT incluida en `LICENSE`.
