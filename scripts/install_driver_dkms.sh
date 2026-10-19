#!/usr/bin/env bash
# Register, build and install the kunlun driver via real DKMS, then load it.
# Requires root (sudo).
#
# Unlike scripts/install_driver.sh (which hand-copies a locally built .ko),
# this registers the driver as an actual DKMS package (kunlun-driver/dkms.conf,
# PACKAGE_NAME/PACKAGE_VERSION from there). Because dkms.conf has AUTOINSTALL=yes,
# a real DKMS install means a future kernel package upgrade auto-rebuilds and
# reinstalls the module via DKMS's kernel-postinst hook — no manual rerun needed.
# scripts/install_driver.sh does NOT give you that; this script exists so you can
# choose durability-across-kernel-upgrades over the simpler manual path.
#
# Builds from the committed state of kunlun-driver/ at HEAD (git archive), not
# the working tree — same as what a real package install would ship. Commit
# driver changes first if you want this script to pick them up.
#
# Design mirrors scripts/install_driver.sh: register/build/install to disk
# first, then try unload/reload; on failure, leave reboot instructions (next
# boot loads the already-installed, dkms-managed .ko).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib_driver_common.sh
source "${REPO_ROOT}/scripts/lib_driver_common.sh"

KVER="$(uname -r)"
MODPROBE_CONF="/etc/modprobe.d/kunlun-kl1.conf"
DISK_ONLY="${DISK_ONLY:-0}"
SKIP_MODPROBE_CONF="${SKIP_MODPROBE_CONF:-0}"
# Set to 1 to let this script `apt-get install dkms` itself if missing.
AUTO_INSTALL_DKMS="${AUTO_INSTALL_DKMS:-0}"

DKMS_CONF="${REPO_ROOT}/kunlun-driver/dkms.conf"
[[ -f "$DKMS_CONF" ]] || die "${DKMS_CONF} not found"

# PACKAGE_NAME/PACKAGE_VERSION come from the repo's own dkms.conf, not hardcoded
# here, so this script can't silently drift from it.
PKG_NAME="$(sed -n 's/^PACKAGE_NAME="\(.*\)"$/\1/p' "$DKMS_CONF")"
PKG_VERSION="$(sed -n 's/^PACKAGE_VERSION="\(.*\)"$/\1/p' "$DKMS_CONF")"
[[ -n "$PKG_NAME" && -n "$PKG_VERSION" ]] || die "could not parse PACKAGE_NAME/PACKAGE_VERSION from ${DKMS_CONF}"
DKMS_SRC_DIR="/usr/src/${PKG_NAME}-${PKG_VERSION}"

if [[ "$(id -u)" -ne 0 ]]; then
    die "run with sudo"
fi

if ! command -v dkms >/dev/null 2>&1; then
    if [[ "$AUTO_INSTALL_DKMS" == "1" ]]; then
        info "dkms not found — installing (AUTO_INSTALL_DKMS=1)"
        apt-get update -qq && apt-get install -y dkms
    else
        die "dkms not installed. Install it (sudo apt-get install dkms) or rerun with AUTO_INSTALL_DKMS=1."
    fi
fi

stage_source() {
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    # Build from the committed tree (HEAD), matching what a real package would
    # ship — not whatever happens to be sitting in the working tree (stray
    # .o/.ko build artifacts, uncommitted edits).
    git -C "$REPO_ROOT" archive HEAD -- kunlun-driver | tar -x -C "$tmp"
    [[ -d "$tmp/kunlun-driver" ]] || die "git archive produced no kunlun-driver/ — run from inside the repo"
    rm -rf "$DKMS_SRC_DIR"
    mkdir -p "$(dirname "$DKMS_SRC_DIR")"
    mv "$tmp/kunlun-driver" "$DKMS_SRC_DIR"
    info "Staged committed kunlun-driver/ (HEAD) -> ${DKMS_SRC_DIR}"
}

dkms_register_build_install() {
    # Drop any prior registration for this name+version first — covers both a
    # stale registration from this script's own earlier run and (if present) a
    # vendor-installed kunlun-module-dkms/4.33.0 pointing at different, unpatched
    # source, which would otherwise silently win on the next kernel upgrade.
    dkms remove -m "$PKG_NAME" -v "$PKG_VERSION" --all >/dev/null 2>&1 || true
    dkms add -m "$PKG_NAME" -v "$PKG_VERSION"
    info "dkms add OK"
    dkms build -m "$PKG_NAME" -v "$PKG_VERSION" -k "$KVER"
    info "dkms build OK"
    dkms install -m "$PKG_NAME" -v "$PKG_VERSION" -k "$KVER" --force
    info "dkms install OK"
}

find_ko_dst() {
    # DKMS install may compress the module (e.g. Ubuntu's kunlun.ko.zst) —
    # match any of the compressed extensions kmod understands, plus plain .ko.
    # Exclude this script's own .bak.* copies of a previous install.
    find "/lib/modules/${KVER}" \( -name 'kunlun.ko' -o -name 'kunlun.ko.zst' \
        -o -name 'kunlun.ko.xz' -o -name 'kunlun.ko.gz' \) -print -quit 2>/dev/null
}

# ─── main ───
info "=== install_driver_dkms: kernel=${KVER} pkg=${PKG_NAME}/${PKG_VERSION} DISK_ONLY=${DISK_ONLY} SKIP_MODPROBE_CONF=${SKIP_MODPROBE_CONF} ==="

if module_is_unloading; then
    warn "Module stuck Unloading — registering/building/installing to disk only, then you must reboot"
    stage_source
    dkms_register_build_install
    depmod -a
    if [[ "$SKIP_MODPROBE_CONF" == "1" ]]; then
        rm -f "$MODPROBE_CONF"
        info "Removed ${MODPROBE_CONF} (stock-friendly)"
    else
        write_modprobe_conf
    fi
    print_reboot_help
    exit 2
fi

stage_source
dkms_register_build_install
depmod -a
info "depmod -a done for ${KVER}"

KO_DST="$(find_ko_dst)"
[[ -n "$KO_DST" ]] || die "dkms install reported success but no kunlun.ko found under /lib/modules/${KVER} — check 'dkms status'"
info "DKMS installed module at: ${KO_DST}"

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
            info "=== install_driver_dkms: OK (hot reload, no kl1 verify) ==="
            modinfo kunlun | grep -E 'filename|srcversion|vermagic' || true
            exit 0
        fi
        verify_loaded "$KO_DST"
        info "=== install_driver_dkms: OK (hot reload) ==="
        info "dkms status:"
        dkms status -m "$PKG_NAME" -v "$PKG_VERSION" || true
        exit 0
    fi
    print_reboot_help
    exit 2
fi

try_load
if [[ "$SKIP_MODPROBE_CONF" == "1" ]]; then
    info "=== install_driver_dkms: OK (cold load, no kl1 verify) ==="
    modinfo kunlun | grep -E 'filename|srcversion|vermagic' || true
    exit 0
fi
verify_loaded "$KO_DST"
info "=== install_driver_dkms: OK (cold load) ==="
info "dkms status:"
dkms status -m "$PKG_NAME" -v "$PKG_VERSION" || true
exit 0
