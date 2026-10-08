#!/usr/bin/env bash
# Cursor Cloud Agent install script for spikenaut-telemetry-etl (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs only what this repo's CI and manifests need:
#   - apt: curl, ca-certificates
#   - uv + Python 3.14
#   - venv with -e ".[dev]"
#   - tool directories exposed to later shells (/etc/profile.d + /usr/local/bin links)
#
# It ends with a dependency fetch/prebuild, not a test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO="env"
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    "$SUDO" apt-get -o Acquire::Retries=5 update -qq
    "$SUDO" env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (curl + CA certs for the installers below) ---
apt_install curl ca-certificates

# --- Python (uv) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  uv_version="0.12.22"
  uv_installer="$(mktemp)"
  trap 'rm -f "$uv_installer"' EXIT
  curl -LsSf "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-installer.sh" -o "$uv_installer"
  printf '%s  %s\n' "58488ae8dbd0773134c92c85e901430e33f99d975bd7f929d26aa9ab0c2f9390" "$uv_installer" |
    sha256sum --check --status
  sh "$uv_installer"
  rm -f "$uv_installer"
  trap - EXIT
fi
# requires-python >=3.14; validate.yml uses 3.14.
uv python install 3.14
if [ ! -x .venv/bin/python ] ||
  ! .venv/bin/python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 14))'; then
  rm -rf .venv
  uv venv --python 3.14 .venv
fi
uv pip install --python .venv/bin/python -e ".[dev]"

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get these directories from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
# In login shells the venv's bin comes first, so python3 is the venv's Python.
# The links skip python and pip so the system python3 stays the default elsewhere.
repo_root="$(pwd)"
tool_dirs=("$repo_root/.venv/bin" "$HOME/.local/bin")
# shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
printf 'export PATH=%q:$PATH\n' "$(IFS=:; echo "${tool_dirs[*]}")" |
  "$SUDO" tee /etc/profile.d/cursor-env-spikenaut-telemetry-etl.sh >/dev/null
for tool in "$repo_root/.venv/bin"/*; do
  name="${tool##*/}"
  case "$name" in
    python* | pip* | activate* | deactivate | Activate.ps1) continue ;;
  esac
  if [ -f "$tool" ] && [ -x "$tool" ]; then
    "$SUDO" ln -sfn "$tool" "/usr/local/bin/$name"
  fi
done
for name in uv uvx python3.14; do
  tool="$HOME/.local/bin/$name"
  if [ -f "$tool" ] && [ -x "$tool" ]; then
    "$SUDO" ln -sfn "$tool" "/usr/local/bin/$name"
  fi
done

echo "Cursor install for spikenaut-telemetry-etl finished."
