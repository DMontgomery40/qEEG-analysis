import json
import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError

def main() -> int:
    base_url = os.getenv("CLIPROXY_BASE_URL", "http://127.0.0.1:8317").rstrip("/")
    api_key = os.getenv("CLIPROXY_API_KEY", "").strip()

    url = f"{base_url}/v1/models"
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        print(f"ERROR: HTTP {e.code} calling {url}", file=sys.stderr)
        return 2
    except URLError as e:
        print(f"ERROR: Cannot reach CLIProxyAPI at {base_url}", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 3

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("ERROR: /v1/models did not return JSON", file=sys.stderr)
        print(raw[:500], file=sys.stderr)
        return 4

    models = data.get("data", [])
    ids = []
    for m in models:
        mid = m.get("id")
        if isinstance(mid, str):
            ids.append(mid)

    print(json.dumps({"base_url": base_url, "count": len(ids), "models": ids}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
