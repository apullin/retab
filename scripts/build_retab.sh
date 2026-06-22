#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

repo_root="$(retab_repo_root)"
venv_arg=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      venv_arg="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--venv PATH]" >&2
      exit 1
      ;;
  esac
done

venv_dir="$(resolve_venv_dir "${repo_root}" "${venv_arg}")"
ensure_venv "${venv_dir}"
python_bin="${venv_dir}/bin/python"
site_packages="$("${python_bin}" - <<'PY'
import site

for path in site.getsitepackages():
    if path.endswith("site-packages"):
        print(path)
        break
else:
    raise SystemExit("no site-packages dir found")
PY
)"

uv pip install --python "${python_bin}" maturin

(
  cd "${repo_root}"
  export VIRTUAL_ENV="${venv_dir}"
  export PATH="${HOME}/.cargo/bin:${venv_dir}/bin:${PATH}"
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
  "${venv_dir}/bin/maturin" develop --release
)

rm -f "${site_packages}/retab-source-root-overlay.pth"
printf '%s\n' "${repo_root}" > "${site_packages}/retab-source-root-overlay.pth"

echo "Built RETAB into ${venv_dir}"
