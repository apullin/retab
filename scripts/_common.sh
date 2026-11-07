#!/usr/bin/env bash
set -euo pipefail

retab_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required but was not found on PATH" >&2
    exit 1
  fi
}

ensure_venv() {
  local venv_dir="$1"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "error: an existing virtual environment is required; no executable Python at ${venv_dir}/bin/python" >&2
    echo "hint: create the environment explicitly with uv venv --python 3.12 PATH, then pass --venv PATH" >&2
    exit 1
  fi
  require_uv
}

resolve_venv_dir() {
  local explicit="${2:-}"
  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}"
  else
    echo "error: no virtual environment was provided." >&2
    echo "hint: pass --venv PATH or activate an existing virtual environment (VIRTUAL_ENV)" >&2
    exit 1
  fi
}
