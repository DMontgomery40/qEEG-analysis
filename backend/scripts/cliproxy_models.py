import os

import httpx


def main() -> None:
    base_url = os.getenv("CLIPROXY_BASE_URL", "http://127.0.0.1:8317").rstrip("/")
    api_key = os.getenv("CLIPROXY_API_KEY", "").strip()

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=20.0)
    r.raise_for_status()
    data = r.json().get("data", [])
    ids = [item.get("id") for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
    for mid in ids:
        print(mid)


if __name__ == "__main__":
    main()

