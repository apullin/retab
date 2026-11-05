#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

repo_root="$(retab_repo_root)"
venv_arg=""
executorch_src=""

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
    *)
      echo "usage: $0 [--venv PATH] [--source EXECUTORCH_CHECKOUT]" >&2
      exit 1
      ;;
  esac
done

venv_dir="$(resolve_venv_dir "${repo_root}" "${venv_arg}")"
executorch_src="$(resolve_executorch_src "${repo_root}" "${executorch_src}")"
ensure_venv "${venv_dir}"
python_bin="${venv_dir}/bin/python"

if [[ ! -d "${executorch_src}" ]]; then
  echo "error: ExecuTorch checkout not found at ${executorch_src}" >&2
  exit 1
fi

torch_packages="${RETAB_TORCH_PACKAGES:-torch torchvision torchao}"
uv pip install --python "${python_bin}" ${torch_packages}

mapfile -t executorch_deps < <("${python_bin}" - <<'PY' "${executorch_src}/pyproject.toml"
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    data = tomllib.load(handle)

for dep in data["project"].get("dependencies", []):
    print(dep)
PY
)

if [[ ${#executorch_deps[@]} -gt 0 ]]; then
  uv pip install --python "${python_bin}" "${executorch_deps[@]}"
fi

arm_extra_packages="${RETAB_ARM_EXTRA_PACKAGES:-ethos-u-vela==5.2.0}"
uv pip install --python "${python_bin}" ${arm_extra_packages}

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

serializer_python_src="${executorch_src}/examples/arm/arm-scratch/tosa-tools/serialization/python"
if [[ ! -d "${serializer_python_src}" ]]; then
  echo "error: local tosa_serializer python sources not found at ${serializer_python_src}" >&2
  echo "hint: bootstrap currently expects an ExecuTorch checkout with examples/arm/arm-scratch/tosa-tools present" >&2
  exit 1
fi

uv pip install --python "${python_bin}" ml-dtypes
printf '%s\n' "${serializer_python_src}" > "${site_packages}/tosa-serializer-source-overlay.pth"
cat > "${site_packages}/tosa_serializer.py" <<'PY'
from enum import IntEnum

from serializer.tosa_serializer import *  # noqa: F401,F403
from serializer.tosa_serializer import TosaSerializer as _PurePythonTosaSerializer
from serializer.tosa_serializer import (
    ConstMode,
    TensorDir,
    TosaSerializerOpLocation as TosaOpLocation,
)
from serializer.tosa_serializer_utils import dtype_str_to_val
from tosa.BlockSize import BlockSize
from tosa.NanPropagationMode import NanPropagationMode
from tosa.ResizeMode import ResizeMode
from tosa.RoundingMode import RoundingMode
from tosa.Version import Version as TosaVersion


class TosaVersionCompat(IntEnum):
    COMPLETELY_COMPATIBLE = 0
    BACKWARD_COMPATIBLE = 1
    NOT_COMPATIBLE = 2


COMPLETELY_COMPATIBLE = TosaVersionCompat.COMPLETELY_COMPATIBLE
BACKWARD_COMPATIBLE = TosaVersionCompat.BACKWARD_COMPATIBLE
NOT_COMPATIBLE = TosaVersionCompat.NOT_COMPATIBLE

EMBED = ConstMode.EMBED
EMBED_DUMP = ConstMode.EMBED_DUMP
INPUTS = ConstMode.INPUTS

PLACEHOLDER = TensorDir.PLACEHOLDER
CONST = TensorDir.CONST
INTERMEDIATE = TensorDir.INTERMEDIATE
RESULT = TensorDir.RESULT

ShapeMap = dict
TensorMap = dict


class TosaSerializer(_PurePythonTosaSerializer):
    def __init__(
        self,
        pathPrefix,
        constMode=ConstMode.EMBED,
        *,
        targetMajor=None,
        targetMinor=None,
        targetPatch=None,
        targetDraft=None,
        tosa_version_major=None,
        tosa_version_minor=None,
        tosa_version_patch=None,
        tosa_version_draft=None,
    ):
        super().__init__(
            pathPrefix,
            constMode=constMode,
            tosa_version_major=(
                targetMajor if targetMajor is not None else tosa_version_major
            ),
            tosa_version_minor=(
                targetMinor if targetMinor is not None else tosa_version_minor
            ),
            tosa_version_patch=(
                targetPatch if targetPatch is not None else tosa_version_patch
            ),
            tosa_version_draft=(
                targetDraft if targetDraft is not None else tosa_version_draft
            ),
        )
PY

printf '%s\n' "${executorch_src}/src" > "${site_packages}/executorch-source-overlay.pth"

"${python_bin}" - <<'PY'
import tosa_serializer
import torchao.prototype.mx_formats  # noqa: F401  ExecuTorch 1.6 imports it with TOSABackend
from executorch.backends.arm.tosa.backend import TOSABackend

if not hasattr(tosa_serializer, "__file__"):
    raise SystemExit("error: tosa_serializer is not importable")
if not callable(getattr(TOSABackend, "_preprocess", None)):
    raise SystemExit("error: TOSABackend._preprocess not found; RETAB cannot patch this ExecuTorch")

optional_seams = []
if hasattr(TOSABackend, "set_preprocess_override"):
    optional_seams.append("TOSABackend.set_preprocess_override")
try:
    from executorch.exir.program import set_edge_manager_for_partitioners_override  # noqa: F401
except ImportError:
    pass
else:
    optional_seams.append("set_edge_manager_for_partitioners_override")
print("ExecuTorch check: ok (optional fork seams: " + (", ".join(optional_seams) or "none") + ")")
PY

echo "Installed ExecuTorch dependencies and source overlay from ${executorch_src} into ${venv_dir}"
