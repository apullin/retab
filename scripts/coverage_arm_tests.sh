#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

usage() {
  cat <<'EOF'
usage: scripts/coverage_arm_tests.sh [options]

Build retab with Rust source-based coverage instrumentation, run the ARM TOSA
pytest lanes, merge .profraw files, and emit text/html/lcov/json coverage reports.

Options:
  --source PATH          ExecuTorch checkout. Defaults to ../../executorch or EXECUTORCH_SRC.
  --venv PATH            Python venv. Defaults to ../../.venv from this repo layout.
  --out PATH             Coverage output dir. Defaults to target/coverage/arm-tests.
  --jobs N|auto          pytest-xdist worker count. Defaults to auto.
  --model-jobs N         pytest-xdist worker count for model lanes. Defaults to 4; max 4.
  --lane NAME            Run one lane. Can be passed more than once.
                         Valid: ops-int, ops-fp, models-int, models-fp.
  --strict-tests         Exit nonzero if any pytest lane exits nonzero.
  --keep-instrumented    Leave the installed extension coverage-instrumented.
  --llvm-profdata PATH   Override llvm-profdata.
  --llvm-cov PATH        Override llvm-cov.
  -h, --help             Show this help.

Default lanes: ops-int, ops-fp, models-int, models-fp.
Known baseline test failures are allowed by default so coverage reports are
still produced. Use --strict-tests for CI-style failure semantics.
EOF
}

find_llvm_tool() {
  local tool="$1"
  local override="${2:-}"
  local candidate=""
  local host=""
  local sysroot=""

  if [[ -n "${override}" ]]; then
    if [[ -x "${override}" ]]; then
      printf '%s\n' "${override}"
      return 0
    fi
    echo "error: ${tool} override is not executable: ${override}" >&2
    exit 1
  fi

  host="$(rustc -vV | awk '/^host:/ { print $2 }')"
  sysroot="$(rustc --print sysroot)"
  candidate="${sysroot}/lib/rustlib/${host}/bin/${tool}"
  if [[ -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  if command -v "${tool}" >/dev/null 2>&1; then
    command -v "${tool}"
    return 0
  fi

  if command -v xcrun >/dev/null 2>&1; then
    candidate="$(xcrun --find "${tool}" 2>/dev/null || true)"
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  echo "error: ${tool} was not found." >&2
  echo "hint: rustup component add llvm-tools-preview, or pass --${tool} PATH" >&2
  exit 1
}

valid_lane() {
  case "$1" in
    ops-int | ops-fp | models-int | models-fp) return 0 ;;
    *) return 1 ;;
  esac
}

lane_pytest_args() {
  local lane="$1"
  local lane_jobs="${jobs}"

  case "${lane}" in
    models-int | models-fp)
      lane_jobs="${model_jobs}"
      ;;
  esac

  case "${lane}" in
    ops-int)
      printf '%s\0' \
        -q \
        "--numprocesses=${lane_jobs}" \
        backends/arm/test/ \
        --ignore=backends/arm/test/models \
        -k \
        "tosa and not FP and (INT or 16a8w or a16w8 or quantized or qat)"
      ;;
    ops-fp)
      printf '%s\0' \
        -q \
        "--numprocesses=${lane_jobs}" \
        backends/arm/test/ \
        --ignore=backends/arm/test/models \
        -k \
        "tosa and FP"
      ;;
    models-int)
      printf '%s\0' \
        -q \
        "--numprocesses=${lane_jobs}" \
        backends/arm/test/models \
        -k \
        "tosa and INT"
      ;;
    models-fp)
      printf '%s\0' \
        -q \
        "--numprocesses=${lane_jobs}" \
        backends/arm/test/models \
        -k \
        "tosa and FP"
      ;;
  esac
}

run_lane() {
  local lane="$1"
  local log_file="${logs_dir}/${lane}.log"
  local status=0
  local args=()

  while IFS= read -r -d '' arg; do
    args+=("${arg}")
  done < <(lane_pytest_args "${lane}")

  {
    printf '==> lane: %s\n' "${lane}"
    printf '==> cwd: %s\n' "${executorch_src}"
    printf '==> command: %q' "${python_bin}"
    printf ' %q' -m pytest "${args[@]}"
    printf '\n'
  } | tee "${log_file}"

  set +e
  (
    cd "${executorch_src}"
    "${python_bin}" -m pytest "${args[@]}"
  ) 2>&1 | tee -a "${log_file}"
  status="${PIPESTATUS[0]}"
  set -e

  printf '%s,%s\n' "${lane}" "${status}" >> "${status_file}"
  if [[ "${status}" -ne 0 ]]; then
    any_test_failure=1
    echo "warning: lane ${lane} exited ${status}; continuing to coverage merge" >&2
  fi
}

repo_root="$(retab_repo_root)"
venv_arg=""
executorch_arg=""
out_dir="${repo_root}/target/coverage/arm-tests"
jobs="auto"
model_jobs="4"
strict_tests=0
keep_instrumented=0
llvm_profdata_arg="${LLVM_PROFDATA:-}"
llvm_cov_arg="${LLVM_COV:-}"
lanes=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      executorch_arg="$2"
      shift 2
      ;;
    --venv)
      venv_arg="$2"
      shift 2
      ;;
    --out)
      out_dir="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --model-jobs)
      model_jobs="$2"
      shift 2
      ;;
    --lane)
      if ! valid_lane "$2"; then
        echo "error: invalid lane: $2" >&2
        usage >&2
        exit 1
      fi
      lanes+=("$2")
      shift 2
      ;;
    --strict-tests)
      strict_tests=1
      shift
      ;;
    --keep-instrumented)
      keep_instrumented=1
      shift
      ;;
    --llvm-profdata)
      llvm_profdata_arg="$2"
      shift 2
      ;;
    --llvm-cov)
      llvm_cov_arg="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${model_jobs}" in
  1 | 2 | 3 | 4) ;;
  *)
    echo "error: --model-jobs must be 1, 2, 3, or 4; got ${model_jobs}" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "${#lanes[@]}" -eq 0 ]]; then
  lanes=(ops-int ops-fp models-int models-fp)
fi

mkdir -p "${out_dir}"
out_dir="$(cd "${out_dir}" && pwd)"

workspace_root="$(cd "${repo_root}/../.." && pwd)"
if [[ -n "${venv_arg}" ]]; then
  venv_dir="${venv_arg}"
elif [[ -x "${workspace_root}/.venv/bin/python" ]]; then
  venv_dir="${workspace_root}/.venv"
else
  venv_dir="$(resolve_venv_dir "${repo_root}" "")"
fi

if [[ -n "${executorch_arg}" ]]; then
  executorch_src="${executorch_arg}"
elif [[ -n "${EXECUTORCH_SRC:-}" ]]; then
  executorch_src="${EXECUTORCH_SRC}"
elif [[ -d "${workspace_root}/executorch" ]]; then
  executorch_src="${workspace_root}/executorch"
else
  executorch_src="$(resolve_executorch_src "${repo_root}" "")"
fi
python_bin="${venv_dir}/bin/python"
target_triple="$(rustc -vV | awk '/^host:/ { print $2 }')"
target_rustflags_env="CARGO_TARGET_$(printf '%s' "${target_triple}" | tr '[:lower:]-' '[:upper:]_')_RUSTFLAGS"

flatc_tool_path="/opt/homebrew/bin:/opt/homebrew/opt/flatbuffers/bin:/usr/local/bin"
while IFS= read -r flatc_candidate; do
  if [[ -x "${flatc_candidate}" ]] && "${flatc_candidate}" --version 2>/dev/null | grep -q ' 24\.'; then
    flatc_tool_path="$(dirname "${flatc_candidate}"):${flatc_tool_path}"
    break
  fi
done < <(find "${executorch_src}/pip-out" -path '*/flatc_ep/bin/flatc' -type f 2>/dev/null | sort)

profraw_dir="${out_dir}/profraw"
build_profraw_dir="${out_dir}/build-profraw"
logs_dir="${out_dir}/logs"
html_dir="${out_dir}/html"
uv_cache_dir="${out_dir}/uv-cache"
cargo_target_dir="${CARGO_TARGET_DIR:-${out_dir}/cargo-target}"
profdata_file="${out_dir}/retab.profdata"
report_file="${out_dir}/summary.txt"
lcov_file="${out_dir}/lcov.info"
json_file="${out_dir}/coverage.json"
status_file="${out_dir}/pytest_status.csv"

rm -rf "${profraw_dir}" "${build_profraw_dir}" "${html_dir}"
if [[ -z "${CARGO_TARGET_DIR:-}" ]]; then
  rm -rf "${cargo_target_dir}"
fi
mkdir -p "${profraw_dir}" "${build_profraw_dir}" "${logs_dir}" "${html_dir}" "${uv_cache_dir}" "${cargo_target_dir}"
rm -f "${profdata_file}" "${report_file}" "${lcov_file}" "${json_file}" "${status_file}"
printf 'lane,exit_status\n' > "${status_file}"
export UV_CACHE_DIR="${uv_cache_dir}"

llvm_profdata="$(find_llvm_tool llvm-profdata "${llvm_profdata_arg}")"
llvm_cov="$(find_llvm_tool llvm-cov "${llvm_cov_arg}")"

ensure_venv "${venv_dir}"
if [[ ! -x "${venv_dir}/bin/maturin" ]]; then
  uv pip install --python "${python_bin}" maturin
fi

echo "==> building instrumented retab"
(
  cd "${repo_root}"
  existing_target_rustflags="$(printenv "${target_rustflags_env}" || true)"
  coverage_rustflags="${RUSTFLAGS:-}"
  export VIRTUAL_ENV="${venv_dir}"
  export PATH="${HOME}/.cargo/bin:${flatc_tool_path}:${venv_dir}/bin:${PATH}"
  export CARGO_INCREMENTAL=0
  export CARGO_TARGET_DIR="${cargo_target_dir}"
  export "${target_rustflags_env}=${existing_target_rustflags:+${existing_target_rustflags} }${coverage_rustflags:+${coverage_rustflags} }-Cinstrument-coverage"
  export LLVM_PROFILE_FILE="${build_profraw_dir}/build-%p-%m.profraw"
  unset RUSTFLAGS CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
  "${venv_dir}/bin/maturin" develop --release --skip-install
)

native_ext="$(PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - <<'PY'
import retab._native
print(retab._native.__file__)
PY
)"
coverage_object="${out_dir}/$(basename "${native_ext}")"
cp "${native_ext}" "${coverage_object}"

export ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING=1
export ARM_TEST_ENABLE_ARM_TOSA_ENGINE_PATCH=1
export ARM_TOSA_ENGINE_SERIALIZER=rust
export ARM_TOSA_ENGINE_ROOT="${repo_root}"
export PYTHONPATH="${repo_root}/python:${executorch_src}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LLVM_PROFILE_FILE="${profraw_dir}/retab-%p-%m.profraw"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER

any_test_failure=0
for lane in "${lanes[@]}"; do
  run_lane "${lane}"
done

profraw_count="$(find "${profraw_dir}" -name '*.profraw' -type f | wc -l | tr -d ' ')"
if [[ "${profraw_count}" -eq 0 ]]; then
  echo "error: no .profraw files were emitted under ${profraw_dir}" >&2
  exit 1
fi

echo "==> merging ${profraw_count} raw profile files"
"${llvm_profdata}" merge -sparse "${profraw_dir}"/*.profraw -o "${profdata_file}"

ignore_regex='/(rustc|\.cargo/registry|vendor)/'

echo "==> writing ${report_file}"
"${llvm_cov}" report "${coverage_object}" \
  -instr-profile="${profdata_file}" \
  -ignore-filename-regex="${ignore_regex}" \
  "${repo_root}/src" | tee "${report_file}"

echo "==> writing ${html_dir}"
"${llvm_cov}" show "${coverage_object}" \
  -instr-profile="${profdata_file}" \
  -ignore-filename-regex="${ignore_regex}" \
  -format=html \
  -output-dir="${html_dir}" \
  "${repo_root}/src" >/dev/null

echo "==> writing ${lcov_file}"
"${llvm_cov}" export "${coverage_object}" \
  -instr-profile="${profdata_file}" \
  -ignore-filename-regex="${ignore_regex}" \
  -format=lcov \
  "${repo_root}/src" > "${lcov_file}"

echo "==> writing ${json_file}"
"${llvm_cov}" export "${coverage_object}" \
  -instr-profile="${profdata_file}" \
  -ignore-filename-regex="${ignore_regex}" \
  -format=text \
  "${repo_root}/src" > "${json_file}"

if [[ "${keep_instrumented}" -eq 0 ]]; then
  echo "==> restoring normal release retab extension"
  (
    cd "${repo_root}"
    export VIRTUAL_ENV="${venv_dir}"
    export PATH="${HOME}/.cargo/bin:${flatc_tool_path}:${venv_dir}/bin:${PATH}"
    export UV_CACHE_DIR="${uv_cache_dir}"
    unset CARGO_TARGET_DIR RUSTFLAGS "${target_rustflags_env}" LLVM_PROFILE_FILE CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
    "${venv_dir}/bin/maturin" develop --release --skip-install
  )
fi

echo "coverage summary: ${report_file}"
echo "coverage html:    ${html_dir}/index.html"
echo "coverage lcov:    ${lcov_file}"
echo "pytest statuses:  ${status_file}"

if [[ "${strict_tests}" -eq 1 && "${any_test_failure}" -ne 0 ]]; then
  exit 1
fi
