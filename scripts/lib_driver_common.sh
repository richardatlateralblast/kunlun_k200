# Shared functions for scripts/install_driver.sh and scripts/install_driver_dkms.sh.
# Not meant to be run directly — sourced after the caller has set:
#   REPO_ROOT, KVER, MODPROBE_CONF
# and (for build_modargs/write_modprobe_conf) the KL1_* env vars.

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN: $*" >&2; }
info() { echo "$*"; }

module_state_line() {
    # e.g. "kunlun 123 - Live" or "kunlun 123 -1 - Unloading"
    grep '^kunlun ' /proc/modules 2>/dev/null || true
}

module_is_unloading() {
    module_state_line | grep -q 'Unloading'
}

module_is_loaded() {
    lsmod | grep -q '^kunlun '
}

build_modargs() {
    MODARGS=()
    [[ "${KL1_P2P_STUB:-0}" == "1" ]] && MODARGS+=(kl1_p2p_stub=1)
    if [[ "${KL1_DMA_DIRECT:-1}" == "1" ]]; then
        MODARGS+=(kl1_dma_direct=1)
    else
        MODARGS+=(kl1_dma_direct=0)
    fi
    if [[ "${KL1_BOUNCE_PIPE:-1}" == "1" ]]; then
        MODARGS+=(kl1_bounce_pipe=1)
    else
        MODARGS+=(kl1_bounce_pipe=0)
    fi
}

write_modprobe_conf() {
    build_modargs
    # Boot-time defaults so reboot picks production knobs without re-running the script.
    {
        echo "# Managed by test_xpu scripts/install_driver*.sh — do not hand-edit lightly"
        echo "# $(date -Is 2>/dev/null || date)"
        if [[ ${#MODARGS[@]} -gt 0 ]]; then
            echo "options kunlun ${MODARGS[*]}"
        else
            echo "# options kunlun (no extra params)"
        fi
    } >"${MODPROBE_CONF}"
    info "Wrote ${MODPROBE_CONF}: options kunlun ${MODARGS[*]:-(none)}"
}

print_reboot_help() {
    cat >&2 <<EOF

=== ACTION REQUIRED: reboot to finish driver switch ===
On-disk module is already the patched build. Do NOT loop modprobe -r
(that can leave kunlun stuck in "Unloading").

  sudo reboot

After boot, verify:

  modinfo -F srcversion,vermagic kunlun
  cat /sys/module/kunlun/parameters/kl1_p2p_stub \\
      /sys/module/kunlun/parameters/kl1_dma_direct \\
      /sys/module/kunlun/parameters/kl1_bounce_pipe
  ls /dev/xpu* | wc -l    # expect 8 + xpuctrl
  cd ${REPO_ROOT} && make regression

If knobs missing after reboot, run this script again.

EOF
}

stop_userspace_holders() {
    # Narrow patterns only — avoid pkill -f that could race with modprobe.
    pkill -x test_p2p 2>/dev/null || true
    pkill -x test_p2p_verify 2>/dev/null || true
    pkill -x test_host_alloc 2>/dev/null || true
    pkill -x test_pageable_verify 2>/dev/null || true
    pkill -x xpu_perf_test 2>/dev/null || true
    # Do not fuser -k /dev/xpuctrl (can disrupt control path mid-unload).
    for n in /dev/xpu0 /dev/xpu1 /dev/xpu2 /dev/xpu3 /dev/xpu4 /dev/xpu5 /dev/xpu6 /dev/xpu7; do
        fuser -k "$n" 2>/dev/null || true
    done
    sleep 1
}

try_unload() {
    local i refcnt line

    if module_is_unloading; then
        warn "kunlun already in Unloading state — cannot recover without reboot"
        return 1
    fi

    if ! module_is_loaded; then
        info "kunlun not loaded"
        return 0
    fi

    stop_userspace_holders
    refcnt="$(cat /sys/module/kunlun/refcnt 2>/dev/null || echo '?')"
    info "kunlun refcnt=${refcnt}"

    for i in 1 2 3; do
        if module_is_unloading; then
            warn "entered Unloading during attempt ${i}"
            return 1
        fi
        # Prefer rmmod; short timeout via background+kill if hang
        if timeout 8 rmmod kunlun 2>/dev/null || timeout 8 modprobe -r kunlun 2>/dev/null; then
            info "kunlun unloaded"
            return 0
        fi
        info "unload attempt ${i}/3 failed, wait..."
        stop_userspace_holders
        sleep 3
    done

    line="$(module_state_line)"
    refcnt="$(cat /sys/module/kunlun/refcnt 2>/dev/null || echo '?')"
    warn "could not unload kunlun (refcnt=${refcnt} state='${line}')"
    return 1
}

try_load() {
    build_modargs
    if [[ ${#MODARGS[@]} -gt 0 ]]; then
        modprobe kunlun "${MODARGS[@]}"
        info "Loaded with ${MODARGS[*]}"
    else
        modprobe kunlun
        info "Loaded kunlun (default params)"
    fi
}

# modinfo_file FIELD PATH — like `modinfo -F FIELD PATH`, but transparently
# decompresses .zst/.xz/.gz first (installed modules are often compressed —
# e.g. Ubuntu's dkms install produces kunlun.ko.zst — and modinfo doesn't
# decompress a path argument itself, only a loaded module looked up by name).
modinfo_file() {
    # modinfo needs a .ko-suffixed path to recognize the file at all, even
    # once it's plain ELF again — a bare mktemp name isn't enough.
    local field="$1" path="$2" tmp
    case "$path" in
        *.zst) tmp="$(mktemp --suffix=.ko)"; zstd -dqf "$path" -o "$tmp" 2>/dev/null && modinfo -F "$field" "$tmp" 2>/dev/null; rm -f "$tmp" ;;
        *.xz)  tmp="$(mktemp --suffix=.ko)"; xz -dc "$path" >"$tmp" 2>/dev/null && modinfo -F "$field" "$tmp" 2>/dev/null; rm -f "$tmp" ;;
        *.gz)  tmp="$(mktemp --suffix=.ko)"; gunzip -c "$path" >"$tmp" 2>/dev/null && modinfo -F "$field" "$tmp" 2>/dev/null; rm -f "$tmp" ;;
        *)     modinfo -F "$field" "$path" 2>/dev/null ;;
    esac
}

# verify_loaded KO_DST [KO_SRC]
# KO_DST: the on-disk module that should now be loaded (after install/depmod/modprobe),
#         possibly compressed (.ko/.ko.zst/.ko.xz/.ko.gz).
# KO_SRC: optional — if given, also sha256-compares it against KO_DST (manual-copy path only;
#         the dkms path has no single uncompressed "source .ko" to compare, so it's omitted there).
verify_loaded() {
    local ko_dst="$1" ko_src="${2:-}"
    local src_ver live_ver p

    live_ver="$(modinfo -F srcversion kunlun 2>/dev/null || true)"

    info "Driver loaded:"
    modinfo kunlun | grep -E 'filename|version|srcversion|vermagic|kl1_p2p|kl1_dma|kl1_bounce' || true

    if [[ -n "$ko_src" ]]; then
        sha256sum "$ko_dst" "$ko_src" 2>/dev/null || true
        src_ver="$(modinfo_file srcversion "$ko_src")"
        if [[ -n "$src_ver" && -n "$live_ver" && "$src_ver" != "$live_ver" ]]; then
            warn "live srcversion ${live_ver} != build ${src_ver} — old image still running; reboot"
            print_reboot_help
            exit 2
        fi
    else
        src_ver="$(modinfo_file srcversion "$ko_dst")"
        if [[ -z "$src_ver" ]]; then
            warn "could not read srcversion from ${ko_dst} — skipping on-disk/live mismatch check"
        elif [[ -n "$live_ver" && "$src_ver" != "$live_ver" ]]; then
            warn "live srcversion ${live_ver} != on-disk ${src_ver} — old image still running; reboot"
            print_reboot_help
            exit 2
        fi
    fi

    for p in kl1_p2p_stub kl1_dma_direct kl1_bounce_pipe; do
        if [[ ! -f /sys/module/kunlun/parameters/${p} ]]; then
            die "${p} sysfs missing — loaded module is not the patched build. Reboot after disk install, or rebuild."
        fi
        chmod 666 "/sys/module/kunlun/parameters/${p}" 2>/dev/null || true
        info "${p}=$(cat /sys/module/kunlun/parameters/${p})"
    done
    for p in kl1_bounce_d2h kl1_pageable_pin; do
        if [[ -f /sys/module/kunlun/parameters/${p} ]]; then
            chmod 666 "/sys/module/kunlun/parameters/${p}" 2>/dev/null || true
            info "${p}=$(cat /sys/module/kunlun/parameters/${p})"
        else
            warn "${p} missing (older module?)"
        fi
    done
}
