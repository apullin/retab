#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

repo_root="$(retab_repo_root)"
venv_arg=""
executorch_src=""
skip_smoke=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      venv_arg="$2"
      shift 2
      ;;
    --source|--executorch-src)
      executorch_src="$2"
      shift 2
      ;;
    --skip-smoke)
      skip_smoke=1
      shift
      ;;
    *)
      echo "usage: $0 [--venv PATH] [--source EXECUTORCH_CHECKOUT] [--skip-smoke]" >&2
      exit 1
      ;;
  esac
done

venv_dir="$(resolve_venv_dir "${repo_root}" "${venv_arg}")"
ensure_venv "${venv_dir}"

"${repo_root}/scripts/install_executorch.sh" \
  --venv "${venv_dir}" \
  --source "$(resolve_executorch_src "${repo_root}" "${executorch_src}")"
"${repo_root}/scripts/build_retab.sh" --venv "${venv_dir}"

if [[ "${skip_smoke}" -eq 0 ]]; then
  "${repo_root}/scripts/smoke_checkout_build_and_lower.sh" --venv "${venv_dir}"
fi

echo
echo "Bootstrap complete."
echo "Use ${venv_dir}/bin/python or 'uv run --python ${venv_dir}/bin/python ...'"
