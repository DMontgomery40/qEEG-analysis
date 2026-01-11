# Project-local CLIProxyAPI

This folder is intended to keep CLIProxyAPI configuration (and optionally logs) **scoped to this project**, so other local projects don’t get mixed up.

## Recommended setup

1. Install CLIProxyAPI (macOS):

```bash
brew install cliproxyapi
```

2. Optional: place a CLIProxyAPI config at:

`./.cli-proxy-api/cliproxyapi.conf`

3. Start the app:

```bash
./start.sh
```

If login is required, use the UI buttons (or run `cliproxyapi --login`).
