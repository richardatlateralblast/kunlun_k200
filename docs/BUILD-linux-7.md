# Building the kernel module on Linux 6.15 / 7.0 with gcc 14+

The driver was written for Ubuntu 22.04 (kernel 6.8, gcc 12/13). This tree carries the changes needed to compile it on kernel `7.0.0-38-generic` with gcc 15.2. Built and checked (vermagic, no unresolved symbols) on a Dell R730, then loaded by hand with `insmod`: it bound to the card, created `/dev/xpu0`, `/dev/xpu1`, `/dev/xpuctrl` and `/proc/xpu`, and logged no warnings. **No workload has been run on the card yet.**

## Build

```
sudo apt install build-essential linux-headers-$(uname -r)

# gcc 15 defaults to C23, which breaks the conftest probes: force gnu11
printf '#!/bin/sh\nexec gcc -std=gnu11 "$@"\n' > ~/gcc11 && chmod +x ~/gcc11

cd kunlun-driver
make clean
rm -rf kunlun_module/conftest/*.h        # force the probes to run again
make modules CC=$HOME/gcc11 -j8
modinfo -F vermagic ./kunlun.ko          # must match uname -r
```

gcc 13 does not work with a 7.0 kernel (it rejects `-fmin-function-alignment`).

## What changed and why

- **Kbuild:** kernel 6.15+ ignores `EXTRA_CFLAGS`, so it is forwarded to `ccflags-y`. The conftest probe flags gain `-std=gnu11 -fms-extensions` (kernel 7.0 needs `-fms-extensions`; without it every probe that includes `pci.h` fails on a `sizeof(struct filename)` assertion and the driver takes old-kernel code paths). `-Wno-error=int-conversion` restores the pre-gcc-14 warning.
- **API renames:** `in_irq` to `in_hardirq`; `del_timer` to `timer_delete`; `hrtimer_init` plus `.function` to `hrtimer_setup`; `nth_page(p, j)` to `p + j` (contiguous allocation); `follow_pfn` to `follow_pfnmap_start/end`.
- **Stubs (untested guesses):** the "function is present" probes are inverted and break on gcc 14+ (implicit declarations are errors), so `phys_to_dma` uses the plain address, `dma_is_direct` is assumed false, `dma_map_ops.map_resource` is treated as absent (peer-DMA check returns false), and the console-lock and `jiffies_to_timespec` branches are disabled. `vm_flags_set`/`vm_flags_clear` are always used (`vm_flags` has been const since 6.3).

A wrong guess in the DMA stubs could hang the host or corrupt memory under load, so load the module by hand first (`sudo insmod kunlun-driver/kunlun.ko kl1_p2p_stub=0 kl1_dma_direct=1 kl1_bounce_pipe=1`, then `dmesg`), not through `scripts/install_driver.sh`. The module is unsigned, so Secure Boot blocks it.
