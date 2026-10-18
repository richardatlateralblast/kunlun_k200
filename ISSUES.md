# ISSUES

Local issues discovered during the K200-on-modern-host work (see `RESEARCH.md`). No GitHub issues exist for this repo yet, so every entry below is link-less and local-only, per `CLAUDE.md`.

## Todo

- Official `paddlepaddle-xpu` wheel silently returns zeroed data for any tensor transfer/op (`paddle.to_tensor(...).numpy()`, elementwise add, matmul all come back zero with no exception; `paddle.randn` raises `XDNN_INVALID_PARAM`). Root-caused to the wheel bundling its own, different-build `libxpuapi.so` than this repo's proven-correct one. Fix not yet attempted (see `RESEARCH.md` "Open questions").
- DMA bandwidth on the kernel-7.0 build is roughly half the repo's originally-documented peak (pageable H2D ~5.35 vs ~10.1 GB/s; `xpu_host_alloc` H2D ~6.29 vs ~12.5 GB/s), despite GEMM compute matching the documented numbers almost exactly. Cause not yet isolated.
- `make clean` run at the repo root recurses into `kunlun-driver/kunlun_module` and deletes tracked git files (the conftest compile-test cache, `.cmd` files). Currently only documented as a hazard in `CLAUDE.md` (recoverable via `git restore kunlun-driver/`), not fixed at the Makefile level.
- Driver module's behavior across `rmmod`/reload, and persistence across a reboot when installed via `scripts/install_driver.sh`/DKMS, is untested (only loaded by hand so far).
- Driver load under Secure Boot with a self-signed module is untested (test host has Secure Boot off).

## Done

- `Makefile` hardcoded a nonexistent external SDK path (`/usr/local/xpu-4.33.0`) and a specific developer's machine path (`/mnt/storage/test_xpu`) as defaults, breaking a fresh clone's build. Fixed: `ROOT` now defaults to the checkout itself, with `XPURT_INC`/`XDNN_INC`/`XDNN_LIB` overridable.
- `libxpurt.so` → `libxpurt.so.1` symlink needed by `-lxpurt` was missing from every clone (SDK only ships the versioned file, and it wasn't LFS-tracked). Fixed: now a Makefile prerequisite (auto-created) and committed directly.
- `git lfs pull` was never run in any checked clone, leaving `xdnn-ubuntu_x86_64/so/libxpuapi.so` and other large files (`data/xpu_denoise_synth.bin`, `paddle_models/inference.pdiparams`, `xdnn-ubuntu_x86_64/lib/libxpuapi_ut.a`) as unfetched Git LFS pointer files, causing silent build/link failures against the wrong (134-byte) file. Fixed: `git lfs install && git lfs pull` now called out in the README Quickstart.
- Driver failed to build on kernel 6.15+/7.0 with gcc 14+ (~225 compile errors). Causes isolated and fixed: `EXTRA_CFLAGS` not forwarded to `ccflags-y`; conftest probes missing `-fms-extensions`/`-std=gnu11` causing silent fallback to 6.8-era code paths; inverted "function present" probes misdetecting several gcc-14-hard-errored functions as present; six genuine kernel API changes patched across 13 files. See `docs/BUILD-linux-7.md`.
</content>
