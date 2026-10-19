#!/usr/bin/env bash
# Install and load the modified kunlun.ko from this repo — manual (non-DKMS) path.
# Requires root (sudo).
#
# This hand-copies a locally built kunlun-driver/kunlun.ko to the module tree
# and manages it directly. It works everywhere, but a kernel upgrade will NOT
# auto-rebuild the module (vermagic mismatch after the upgrade) — you have to
# rerun `make driver && make driver-install` by hand. If you want the module
# to survive a kernel upgrade automatically, use scripts/install_driver_dkms.sh
# instead, which registers it with real DKMS.
#
# Design (robust path):
#   1) Always install .ko to disk + depmod + modprobe.d options FIRST
#   2) Then try unload/reload to pick up the new image
#   3) If unload fails / module stuck "Unloading", exit with reboot instructions
#      (next boot loads the already-installed patched .ko)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib_driver_common.sh
source "${REPO_ROOT}/scripts/lib_driver_common.sh"

# Override: KO_SRC=/path/to/kunlun.ko (e.g. restore stock)
# Or create ${REPO_ROOT}/.force_ko_src with a single path line (for sudo make without env).
FORCE_KO_FILE="${REPO_ROOT}/.force_ko_src"
if [[ -z "${KO_SRC:-}" && -f "$FORCE_KO_FILE" ]]; then
    KO_SRC="$(head -1 "$FORCE_KO_FILE" | tr -d '\r' | xargs)"
    # Auto stock-friendly flags unless explicitly set
    DISK_ONLY="${DISK_ONLY:-1}"
    SKIP_MODPROBE_CONF="${SKIP_MODPROBE_CONF:-1}"
    info_force=1
fi
KO_SRC="${KO_SRC:-${REPO_ROOT}/kunlun-driver/kunlun.ko}"
KVER="$(uname -r)"
KO_DST="/lib/modules/${KVER}/updates/dkms/kunlun.ko"
MODPROBE_CONF="/etc/modprobe.d/kunlun-kl1.conf"
# If 1, only write disk artifacts; never unload (use before planned reboot).
DISK_ONLY="${DISK_ONLY:-0}"
# If 1, skip writing kl1_* options (for stock modules without those params).
SKIP_MODPROBE_CONF="${SKIP_MODPROBE_CONF:-0}"

if [[ ! -f "$KO_SRC" ]]; then
    die "${KO_SRC} not found. Run: make driver"
fi

if [[ "$(id -u)" -ne 0 ]]; then
    die "run with sudo"
fi

install_ko_to_disk() {
    local src_ver dst_ver

    src_ver="$(modinfo -F srcversion "$KO_SRC" 2>/dev/null || true)"
    if [[ -f "$KO_DST" ]]; then
        dst_ver="$(modinfo -F srcversion "$KO_DST" 2>/dev/null || true)"
        if [[ -n "$src_ver" && -n "$dst_ver" && "$src_ver" != "$dst_ver" ]]; then
            info "Replacing on-disk ko srcversion ${dst_ver} -> ${src_ver}"
        fi
    fi

    mkdir -p "$(dirname "$KO_DST")"

    if [[ ! -f "${KO_DST}.bak.20260612" && -f "$KO_DST" ]]; then
        cp -a "$KO_DST" "${KO_DST}.bak.20260612"
        info "Backed up original to ${KO_DST}.bak.20260612"
    fi
    # Always keep a timestamped backup of whatever we overwrite (best-effort).
    if [[ -f "$KO_DST" ]]; then
        cp -a "$KO_DST" "${KO_DST}.bak.$(date +%Y%m%d%H%M%S)" 2>/dev/null || true
    fi

    cp -a "$KO_SRC" "$KO_DST"
    info "Installed ${KO_SRC} -> ${KO_DST}"
    depmod -a
    info "depmod -a done for ${KVER}"
}

# ─── main ───
info "=== install_driver (manual, non-DKMS): kernel=${KVER} DISK_ONLY=${DISK_ONLY} SKIP_MODPROBE_CONF=${SKIP_MODPROBE_CONF} ==="
info "KO_SRC=${KO_SRC}"
if [[ -f "$FORCE_KO_FILE" ]]; then
    info "Consumed ${FORCE_KO_FILE}"
    rm -f "$FORCE_KO_FILE"
fi

if module_is_unloading; then
    warn "Module stuck Unloading — install .ko to disk only, then you must reboot"
    install_ko_to_disk
    if [[ "$SKIP_MODPROBE_CONF" == "1" ]]; then
        rm -f "$MODPROBE_CONF"
        info "Removed ${MODPROBE_CONF} (stock-friendly)"
    else
        write_modprobe_conf
    fi
    print_reboot_help
    exit 2
fi

install_ko_to_disk
if [[ "$SKIP_MODPROBE_CONF" == "1" ]]; then
    if [[ -f "$MODPROBE_CONF" ]]; then
        rm -f "$MODPROBE_CONF"
        info "Removed ${MODPROBE_CONF} (SKIP_MODPROBE_CONF=1)"
    fi
else
    write_modprobe_conf
fi

if [[ "$DISK_ONLY" == "1" ]]; then
    info "DISK_ONLY=1 — skipping unload/load. Reboot when ready."
    print_reboot_help
    exit 0
fi

if module_is_loaded; then
    if try_unload; then
        try_load
        if [[ "$SKIP_MODPROBE_CONF" == "1" ]]; then
            info "=== install_driver: OK (hot reload, no kl1 verify) ==="
            modinfo kunlun | grep -E 'filename|srcversion|vermagic' || true
            exit 0
        fi
        verify_loaded "$KO_DST" "$KO_SRC"
        info "=== install_driver: OK (hot reload) ==="
        exit 0
    fi
    # Disk already has new ko; reboot will load it.
    print_reboot_help
    exit 2
fi

# Not loaded
try_load
if [[ "$SKIP_MODPROBE_CONF" == "1" ]]; then
    info "=== install_driver: OK (cold load, no kl1 verify) ==="
    modinfo kunlun | grep -E 'filename|srcversion|vermagic' || true
    exit 0
fi
verify_loaded "$KO_DST" "$KO_SRC"
info "=== install_driver: OK (cold load) ==="
exit 0
