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
  case "$1" in
    backend|pipeline_worker)
      local project_root pid process_root
      project_root="$(cd "$2" && pwd -P)" || return 1
      for pid in $(pgrep -f -- "$pattern" 2>/dev/null); do
        process_root="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
        [ -n "$process_root" ] || continue
        process_root="$(cd "$process_root" 2>/dev/null && pwd -P)" || continue
        [ "$process_root" = "$project_root" ] && return 0
      done
      return 1
      ;;
    *) pgrep -f -- "$pattern" >/dev/null 2>&1 ;;
  esac
}
