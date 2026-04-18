#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

repo_root="$(retab_repo_root)"
venv_arg=""
out_file=""
model="tiny_cnn"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      venv_arg="$2"
      shift 2
      ;;
    --out)
      out_file="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--venv PATH] [--out FILE] [--model NAME]" >&2
      exit 1
      ;;
  esac
done

venv_dir="$(resolve_venv_dir "${repo_root}" "${venv_arg}")"
ensure_venv "${venv_dir}"
python_bin="${venv_dir}/bin/python"

if [[ -z "${out_file}" ]]; then
  out_file="${repo_root}/tmp/${model}.pte"
fi

mkdir -p "$(dirname "${out_file}")"
"${python_bin}" "${repo_root}/examples/lower_with_retab.py" --model "${model}" --out "${out_file}"

if [[ ! -s "${out_file}" ]]; then
  echo "error: smoke lower did not produce a non-empty ${out_file}" >&2
  exit 1
fi

echo "Smoke lower wrote ${out_file}"
