import json
import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError

def _get(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        return e.code, str(e)
    except URLError as e:
        return 0, str(e)

def main() -> int:
    backend = os.getenv("QEEG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    url = f"{backend}/api/health"
    code, body = _get(url)

    if code == 0:
        print(f"ERROR: cannot reach backend at {backend}", file=sys.stderr)
        print(body, file=sys.stderr)
        return 2

    print(f"GET {url} -> {code}")
    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2))
    except Exception:
        print(body[:500])

    return 0 if 200 <= code < 300 else 3

if __name__ == "__main__":
    raise SystemExit(main())
