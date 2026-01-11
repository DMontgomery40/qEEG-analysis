#!/bin/bash

# qEEG Council - Start script

echo "Starting qEEG Council..."
echo ""

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

CLIPROXY_BASE_URL="${CLIPROXY_BASE_URL:-http://127.0.0.1:8317}"
CLIPROXY_HEALTH_URL="${CLIPROXY_BASE_URL%/}/v1/models"

CLIPROXY_PID=""

if [ ! -f ".cli-proxy-api/cliproxyapi.conf" ]; then
  mkdir -p ".cli-proxy-api"
  cat > ".cli-proxy-api/cliproxyapi.conf" <<'EOF'
host: "127.0.0.1"
port: 8317
auth-dir: ".cli-proxy-api/auth"
api-keys: []
EOF
fi

mkdir -p ".cli-proxy-api/auth"
if ls "$HOME/.cli-proxy-api/"*.json >/dev/null 2>&1; then
  for f in "$HOME/.cli-proxy-api/"*.json; do
    base="$(basename "$f")"
    if [ ! -f ".cli-proxy-api/auth/$base" ]; then
      cp "$f" ".cli-proxy-api/auth/$base" >/dev/null 2>&1 || true
    fi
  done
fi

PROJECT_CLIPROXY_CONFIG=".cli-proxy-api/cliproxyapi.conf"
BREW_CLIPROXY_CONFIG="/opt/homebrew/etc/cliproxyapi.conf"

# Prefer a valid CLIPROXY_CONFIG if provided, otherwise use the project-local config,
# otherwise fall back to Homebrew's config.
if [ -n "${CLIPROXY_CONFIG:-}" ] && [ -f "$CLIPROXY_CONFIG" ]; then
  : # keep it
elif [ -f "$PROJECT_CLIPROXY_CONFIG" ]; then
  CLIPROXY_CONFIG="$PROJECT_CLIPROXY_CONFIG"
elif [ -f "$BREW_CLIPROXY_CONFIG" ]; then
  CLIPROXY_CONFIG="$BREW_CLIPROXY_CONFIG"
else
  CLIPROXY_CONFIG=""
fi

CLIPROXY_BIN="$(command -v cliproxyapi 2>/dev/null || true)"
if [ -z "$CLIPROXY_BIN" ] && [ -x "/opt/homebrew/bin/cliproxyapi" ]; then
  CLIPROXY_BIN="/opt/homebrew/bin/cliproxyapi"
fi

if [ -z "$CLIPROXY_BIN" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "CLIProxyAPI not found; installing via Homebrew..."
    brew install cliproxyapi >/tmp/cliproxy_install.log 2>&1 || {
      echo "brew install cliproxyapi failed; attempting tap + install..."
      brew tap router-for-me/tap >>/tmp/cliproxy_install.log 2>&1 || true
      brew install cliproxyapi >>/tmp/cliproxy_install.log 2>&1 || true
    }
  fi
  CLIPROXY_BIN="$(command -v cliproxyapi 2>/dev/null || true)"
  if [ -z "$CLIPROXY_BIN" ] && [ -x "/opt/homebrew/bin/cliproxyapi" ]; then
    CLIPROXY_BIN="/opt/homebrew/bin/cliproxyapi"
  fi
fi

# Optional OCR dependency for PDFs with image-only pages (helps capture tables/metrics embedded as images).
if ! command -v tesseract >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Tesseract not found; installing via Homebrew (for PDF OCR)..."
    brew install tesseract >/tmp/tesseract_install.log 2>&1 || true
  fi
fi

HTTP_CODE="$(curl -s -o /dev/null -w "%{http_code}" "$CLIPROXY_HEALTH_URL" || true)"
if [ "$HTTP_CODE" = "200" ]; then
  echo "✓ CLIProxyAPI is reachable at $CLIPROXY_BASE_URL"
else
  LISTEN_PID="$(lsof -nP -iTCP:8317 -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)"
  if [ -n "$LISTEN_PID" ]; then
    COMM="$(ps -p "$LISTEN_PID" -o comm= 2>/dev/null | tr -d ' ' || true)"
    if echo "$COMM" | grep -q "cliproxyapi"; then
      echo "Stopping existing CLIProxyAPI on :8317 (pid $LISTEN_PID)..."
      kill "$LISTEN_PID" 2>/dev/null || true
      sleep 1
    fi
  fi
  if [ -n "$CLIPROXY_BIN" ]; then
    if [ -n "$CLIPROXY_CONFIG" ] && [ -f "$CLIPROXY_CONFIG" ]; then
      echo "Starting CLIProxyAPI (config: $CLIPROXY_CONFIG)..."
      "$CLIPROXY_BIN" -config "$CLIPROXY_CONFIG" >/tmp/cliproxyapi.log 2>&1 &
      CLIPROXY_PID=$!
      sleep 2
      if curl -sSf "$CLIPROXY_HEALTH_URL" >/dev/null 2>&1; then
        echo "✓ CLIProxyAPI started at $CLIPROXY_BASE_URL"
      else
        echo "⚠ CLIProxyAPI did not become ready. Last log lines:"
        tail -n 30 /tmp/cliproxyapi.log 2>/dev/null || true
        echo ""
        echo "If login is required, run one of:"
        echo "  cliproxyapi -login"
        echo "  cliproxyapi -claude-login"
        echo "  cliproxyapi -codex-login"
        echo ""
      fi
    else
      echo "Starting CLIProxyAPI (no config available; using defaults)..."
      "$CLIPROXY_BIN" >/tmp/cliproxyapi.log 2>&1 &
      CLIPROXY_PID=$!
      sleep 2
      if curl -sSf "$CLIPROXY_HEALTH_URL" >/dev/null 2>&1; then
        echo "✓ CLIProxyAPI started at $CLIPROXY_BASE_URL"
      else
        echo "⚠ CLIProxyAPI did not become ready. Last log lines:"
        tail -n 30 /tmp/cliproxyapi.log 2>/dev/null || true
        echo ""
        echo "If login is required, run:"
        echo "  cliproxyapi -login"
        echo ""
      fi
    fi
  else
    echo "⚠ CLIProxyAPI not reachable and 'cliproxyapi' not found on PATH."
    echo "Install/start CLIProxyAPI, then refresh the UI."
    echo ""
  fi
fi

# Start backend
echo "Starting backend on http://localhost:8000..."
uv run python -m backend.main &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 2

# Start frontend
echo "Starting frontend on http://localhost:5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✓ qEEG Council is running!"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID $CLIPROXY_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
