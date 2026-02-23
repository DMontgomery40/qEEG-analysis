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
PROJECT_ROOT="$(pwd)"
DEFAULT_THRYLEN_REPO="$PROJECT_ROOT/../thrylen"
QEEG_PORTAL_AUTO_SYNC="${QEEG_PORTAL_AUTO_SYNC:-1}"
QEEG_PORTAL_SYNC_DIR="${QEEG_PORTAL_SYNC_DIR:-$PROJECT_ROOT/data/portal_patients}"
QEEG_PORTAL_SYNC_REPO="${QEEG_PORTAL_SYNC_REPO:-$DEFAULT_THRYLEN_REPO}"

CLIPROXY_PID=""
PORTAL_SYNC_PID=""
export PATH="$HOME/.local/bin:$PATH"

find_cliproxy_bin() {
  if command -v cli-proxy-api-plus >/dev/null 2>&1; then
    command -v cli-proxy-api-plus
    return 0
  fi
  if command -v cliproxyapi >/dev/null 2>&1; then
    command -v cliproxyapi
    return 0
  fi
  if [ -x "$HOME/.local/bin/cli-proxy-api-plus" ]; then
    echo "$HOME/.local/bin/cli-proxy-api-plus"
    return 0
  fi
  if [ -x "/opt/homebrew/bin/cli-proxy-api-plus" ]; then
    echo "/opt/homebrew/bin/cli-proxy-api-plus"
    return 0
  fi
  if [ -x "/opt/homebrew/bin/cliproxyapi" ]; then
    echo "/opt/homebrew/bin/cliproxyapi"
    return 0
  fi
  return 1
}

install_cliproxy_plus() {
  local os_name arch_name asset_url tmp_dir
  os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch_name="$(uname -m)"
  case "$arch_name" in
    arm64|aarch64) arch_name="arm64" ;;
    x86_64|amd64) arch_name="amd64" ;;
  esac

  asset_url="$(curl -fsSL 'https://api.github.com/repos/router-for-me/CLIProxyAPIPlus/releases/latest' \
    | grep -Eo "https://[^\"]*CLIProxyAPIPlus_[^\"]*_${os_name}_${arch_name}\\.tar\\.gz" \
    | head -n 1 || true)"
  if [ -z "$asset_url" ]; then
    return 1
  fi

  tmp_dir="$(mktemp -d)"
  curl -fsSL "$asset_url" -o "$tmp_dir/cliproxy-plus.tar.gz" || return 1
  tar -xzf "$tmp_dir/cliproxy-plus.tar.gz" -C "$tmp_dir" || return 1

  mkdir -p "$HOME/.local/bin"
  if [ ! -f "$tmp_dir/cli-proxy-api-plus" ]; then
    return 1
  fi
  install -m 0755 "$tmp_dir/cli-proxy-api-plus" "$HOME/.local/bin/cli-proxy-api-plus" || return 1
  ln -sf "$HOME/.local/bin/cli-proxy-api-plus" "$HOME/.local/bin/cliproxyapi" || true
  return 0
}

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
PROJECT_CLIPROXY_CONFIG_YAML=".cli-proxy-api/config.yaml"
BREW_CLIPROXY_CONFIG="/opt/homebrew/etc/cliproxyapi.conf"

# Prefer a valid CLIPROXY_CONFIG if provided, otherwise use the project-local config,
# otherwise fall back to Homebrew's config.
if [ -n "${CLIPROXY_CONFIG:-}" ] && [ -f "$CLIPROXY_CONFIG" ]; then
  : # keep it
elif [ -f "$PROJECT_CLIPROXY_CONFIG" ]; then
  CLIPROXY_CONFIG="$PROJECT_CLIPROXY_CONFIG"
elif [ -f "$PROJECT_CLIPROXY_CONFIG_YAML" ]; then
  CLIPROXY_CONFIG="$PROJECT_CLIPROXY_CONFIG_YAML"
elif [ -f "$BREW_CLIPROXY_CONFIG" ]; then
  CLIPROXY_CONFIG="$BREW_CLIPROXY_CONFIG"
else
  CLIPROXY_CONFIG=""
fi

CLIPROXY_BIN="$(find_cliproxy_bin || true)"

if [ -z "$CLIPROXY_BIN" ]; then
  echo "CLIProxyAPI Plus not found; installing latest release..."
  if ! install_cliproxy_plus >/tmp/cliproxy_install.log 2>&1; then
    echo "CLIProxyAPI Plus install failed; trying Homebrew cliproxyapi fallback..."
    if command -v brew >/dev/null 2>&1; then
      brew install cliproxyapi >>/tmp/cliproxy_install.log 2>&1 || {
        echo "brew install cliproxyapi failed; attempting tap + install..."
        brew tap router-for-me/tap >>/tmp/cliproxy_install.log 2>&1 || true
        brew install cliproxyapi >>/tmp/cliproxy_install.log 2>&1 || true
      }
    fi
  fi
  CLIPROXY_BIN="$(find_cliproxy_bin || true)"
fi

if [ -z "$CLIPROXY_BIN" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "CLIProxyAPI/Plus still not found. Check /tmp/cliproxy_install.log"
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
  echo "✓ CLIProxyAPI/Plus is reachable at $CLIPROXY_BASE_URL"
else
  LISTEN_PID="$(lsof -nP -iTCP:8317 -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)"
  if [ -n "$LISTEN_PID" ]; then
    COMM="$(ps -p "$LISTEN_PID" -o comm= 2>/dev/null | tr -d ' ' || true)"
    if echo "$COMM" | grep -Eq "cliproxyapi|cli-proxy-api-plus"; then
      echo "Stopping existing CLIProxyAPI/Plus on :8317 (pid $LISTEN_PID)..."
      kill "$LISTEN_PID" 2>/dev/null || true
      sleep 1
    fi
  fi
  if [ -n "$CLIPROXY_BIN" ]; then
    PROXY_CMD="$(basename "$CLIPROXY_BIN")"
    if [ -n "$CLIPROXY_CONFIG" ] && [ -f "$CLIPROXY_CONFIG" ]; then
      echo "Starting CLIProxyAPI/Plus (config: $CLIPROXY_CONFIG)..."
      "$CLIPROXY_BIN" -config "$CLIPROXY_CONFIG" >/tmp/cliproxyapi.log 2>&1 &
      CLIPROXY_PID=$!
      sleep 2
      if curl -sSf "$CLIPROXY_HEALTH_URL" >/dev/null 2>&1; then
        echo "✓ CLIProxyAPI/Plus started at $CLIPROXY_BASE_URL"
      else
        echo "⚠ CLIProxyAPI/Plus did not become ready. Last log lines:"
        tail -n 30 /tmp/cliproxyapi.log 2>/dev/null || true
        echo ""
        echo "If login is required, run one of:"
        echo "  $PROXY_CMD -login"
        echo "  $PROXY_CMD -claude-login"
        echo "  $PROXY_CMD -codex-login"
        echo ""
      fi
    else
      echo "Starting CLIProxyAPI/Plus (no config available; using defaults)..."
      "$CLIPROXY_BIN" >/tmp/cliproxyapi.log 2>&1 &
      CLIPROXY_PID=$!
      sleep 2
      if curl -sSf "$CLIPROXY_HEALTH_URL" >/dev/null 2>&1; then
        echo "✓ CLIProxyAPI/Plus started at $CLIPROXY_BASE_URL"
      else
        echo "⚠ CLIProxyAPI/Plus did not become ready. Last log lines:"
        tail -n 30 /tmp/cliproxyapi.log 2>/dev/null || true
        echo ""
        echo "If login is required, run:"
        echo "  $PROXY_CMD -login"
        echo ""
      fi
    fi
  else
    echo "⚠ CLIProxyAPI/Plus not reachable and no proxy binary found on PATH."
    echo "Install/start CLIProxyAPI Plus (or cliproxyapi), then refresh the UI."
    echo ""
  fi
fi

# Optional: keep Netlify portal blobs in sync while qEEG Council is running.
if [ "$QEEG_PORTAL_AUTO_SYNC" != "0" ]; then
  if [ -d "$QEEG_PORTAL_SYNC_REPO" ] && [ -f "$QEEG_PORTAL_SYNC_REPO/scripts/qeeg_patients_watch.mjs" ]; then
    if command -v npm >/dev/null 2>&1; then
      echo "Starting portal auto-sync watcher..."
      (
        cd "$QEEG_PORTAL_SYNC_REPO" || exit 1
        npm run qeeg:patients:watch -- --dir "$QEEG_PORTAL_SYNC_DIR"
      ) >/tmp/qeeg_patients_watch.log 2>&1 &
      PORTAL_SYNC_PID=$!
      sleep 1
      if ps -p "$PORTAL_SYNC_PID" >/dev/null 2>&1; then
        echo "✓ Portal watcher started (pid $PORTAL_SYNC_PID)"
      else
        echo "⚠ Portal watcher failed to start. See /tmp/qeeg_patients_watch.log"
      fi
    else
      echo "⚠ npm not found; skipping portal auto-sync watcher."
    fi
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
trap "kill $BACKEND_PID $FRONTEND_PID $CLIPROXY_PID $PORTAL_SYNC_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
