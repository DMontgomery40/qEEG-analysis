#!/bin/bash

qeeg_component_pattern() {
  local component="$1"
  local project_root="$2"
  local sync_dir="$3"

  case "$component" in
    portal_watcher)
      printf '%s\n' "qeeg_patients_watch.mjs --dir $sync_dir"
      ;;
    pipeline_worker)
      printf '%s\n' "scripts/portal_pipeline_worker.py --poll-seconds"
      ;;
    backend)
      printf '%s\n' "-m backend.main"
      ;;
    frontend)
      printf '%s\n' "$project_root/frontend/node_modules/.bin/vite"
      ;;
    *)
      return 2
      ;;
  esac
}

qeeg_component_is_running() {
  local pattern
  pattern="$(qeeg_component_pattern "$1" "$2" "$3")" || return 2
  pgrep -f -- "$pattern" >/dev/null 2>&1
}
