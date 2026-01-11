import os

import httpx


def main() -> None:
    base_url = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    r = httpx.get(f"{base_url}/api/health", timeout=20.0)
    r.raise_for_status()
    print(r.json())


if __name__ == "__main__":
    main()

