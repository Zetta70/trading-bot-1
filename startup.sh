#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# startup.sh — Launch the KIS Paper Trading Bot
#
# Usage:
#   chmod +x startup.sh
#   ./startup.sh            # foreground (Ctrl+C to stop)
#   ./startup.sh &          # background
#   nohup ./startup.sh &    # background, survives logout
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python Environment ──────────────────────────────────────────────
# Activate venv if it exists, otherwise use system Python
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ── Pre-flight Checks ──────────────────────────────────────────────
echo "[startup] Checking Python..."
python3 --version

echo "[startup] Installing / updating dependencies..."
pip install -q -r requirements.txt

echo "[startup] Ensuring log directory exists..."
mkdir -p logs

# ── Environment File ───────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "[startup] WARNING: .env not found. Copying from .env.example..."
    cp .env.example .env
    echo "[startup] Edit .env with your settings before running in REAL mode."
fi

# ── Launch ─────────────────────────────────────────────────────────
echo "[startup] Starting trading bot (PID $$)..."
echo "[startup] Logs → logs/system.log, logs/trade.log, logs/error.log"
echo "───────────────────────────────────────────────────────────────"

exec python3 main.py
