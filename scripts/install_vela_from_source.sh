#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

repo_root="$(retab_repo_root)"
venv_arg=""
vela_src=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      venv_arg="$2"
      shift 2
      ;;
    --source|--vela-source)
      vela_src="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--venv PATH] [--source ETHOS_U_VELA_CHECKOUT]" >&2
      exit 1
      ;;
  esac
done

venv_dir="$(resolve_venv_dir "${repo_root}" "${venv_arg}")"
vela_src="$(resolve_vela_src "${repo_root}" "${vela_src}")"
ensure_venv "${venv_dir}"
python_bin="${venv_dir}/bin/python"

if [[ ! -d "${vela_src}" ]]; then
  echo "error: ethos-u-vela checkout not found at ${vela_src}" >&2
  exit 1
fi

uv pip install --python "${python_bin}" \
  "flatbuffers==24.3.25" \
  "numpy>=1.23.0" \
  "setuptools>=82.0.0" \
  "wheel>=0.46.2" \
  "setuptools_scm[toml]~=7.1.0" \
  "cmake>=3.16.3" \
  build

uv pip install --python "${python_bin}" \
  --reinstall \
  --no-build-isolation \
  --editable "${vela_src}[dev]"

"${python_bin}" - <<'PY' "${vela_src}"
import pathlib
import sys

from ethosu import regor
from ethosu.vela import vela

source_root = pathlib.Path(sys.argv[1]).resolve()
vela_path = pathlib.Path(vela.__file__).resolve()
regor_path = pathlib.Path(regor.__file__).resolve()

print(f"vela_python={vela_path}")
print(f"regor_module={regor_path}")

if source_root not in vela_path.parents:
    raise SystemExit(
        "error: editable ethos-u-vela source install is not active for Python modules"
    )
PY

echo "Installed ethos-u-vela from source checkout ${vela_src} into ${venv_dir}"
