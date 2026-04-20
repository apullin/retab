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
  require_uv
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    uv venv "${venv_dir}"
  fi
}

resolve_venv_dir() {
  local repo_root="$1"
  local explicit="${2:-}"
  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
  else
    printf '%s\n' "${repo_root}/.venv"
  fi
}

resolve_executorch_src() {
  local repo_root="$1"
  local explicit="${2:-}"
  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return 0
  fi

  if [[ -n "${EXECUTORCH_SRC:-}" ]]; then
    printf '%s\n' "${EXECUTORCH_SRC}"
    return 0
  fi

  if [[ -d "${repo_root}/../executorch" ]]; then
    printf '%s\n' "${repo_root}/../executorch"
    return 0
  fi

  echo "error: no ExecuTorch checkout was provided." >&2
  echo "hint: pass --source /path/to/executorch or set EXECUTORCH_SRC" >&2
  echo "hint: current compatibility notes are in compat/executorch.toml" >&2
  exit 1
}

resolve_vela_src() {
  local repo_root="$1"
  local explicit="${2:-}"
  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return 0
  fi

  if [[ -n "${VELA_SRC:-}" ]]; then
    printf '%s\n' "${VELA_SRC}"
    return 0
  fi

  if [[ -d "${repo_root}/../ethos-u-vela" ]]; then
    printf '%s\n' "${repo_root}/../ethos-u-vela"
    return 0
  fi

  echo "error: no ethos-u-vela checkout was provided." >&2
  echo "hint: pass --vela-source /path/to/ethos-u-vela or set VELA_SRC" >&2
  exit 1
}
