# ROOT defaults to this checkout, so `make all` works from a fresh clone with
# no other setup. Override ROOT (or XPURT_INC/XDNN_INC/XDNN_LIB individually)
# to build against a system-installed XPURT SDK instead, e.g.:
#   make XPURT_INC=/usr/local/xpu-4.33.0/include XDNN_LIB=/usr/local/xpu-4.33.0/lib64
ROOT = $(CURDIR)
CXX = g++
# xpu/runtime.h (the low-level XPURT API) lives in kunlun-driver/include, not
# in the xdnn-ubuntu_x86_64 SDK tree; xdnn.h (the compute API) is the other way
# round. Both are bundled in this repo, so no external SDK install is needed.
XPURT_INC = $(ROOT)/kunlun-driver/include
XDNN_INC = $(ROOT)/xdnn-ubuntu_x86_64/include
XDNN_LIB = $(ROOT)/xdnn-ubuntu_x86_64/so
CXXFLAGS = -std=c++17 -O2 -I$(XDNN_INC) -I$(XPURT_INC)
LDFLAGS = -L$(XDNN_LIB) -lxpuapi -lxpurt -Wl,-rpath,$(XDNN_LIB)

BENCHMARKS = benchmarks/xpu_perf_test benchmarks/xpu_denoise benchmarks/xpu_int8_probe \
	benchmarks/xpu_app_pipeline benchmarks/xpu_pipeline_p2p
TESTS = tests/test_p2p tests/test_p2p_verify tests/test_host_alloc tests/test_pageable_verify

# The SDK ships libxpurt.so.1 but the linker's -lxpurt needs an unversioned
# libxpurt.so symlink. Git LFS does not store it (it's not tracked content,
# just a build-time link), so every benchmark/test target depends on it.
$(XDNN_LIB)/libxpurt.so:
	ln -sf libxpurt.so.1 $@

all: $(BENCHMARKS) $(TESTS)

benchmarks/xpu_perf_test: benchmarks/xpu_perf_test.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

benchmarks/xpu_denoise: benchmarks/xpu_denoise.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

benchmarks/xpu_int8_probe: benchmarks/xpu_int8_probe.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

benchmarks/xpu_app_pipeline: benchmarks/xpu_app_pipeline.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

benchmarks/xpu_pipeline_p2p: benchmarks/xpu_pipeline_p2p.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

tests/test_p2p: tests/test_p2p.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

tests/test_p2p_verify: tests/test_p2p_verify.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

tests/test_host_alloc: tests/test_host_alloc.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

tests/test_pageable_verify: tests/test_pageable_verify.cpp | $(XDNN_LIB)/libxpurt.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

driver:
	$(MAKE) -C kunlun-driver modules

driver-install: driver
	sudo KL1_P2P_STUB=$${KL1_P2P_STUB:-0} \
		KL1_DMA_DIRECT=$${KL1_DMA_DIRECT:-1} \
		KL1_BOUNCE_PIPE=$${KL1_BOUNCE_PIPE:-1} \
		DISK_ONLY=$${DISK_ONLY:-0} \
		scripts/install_driver.sh

# Install without rebuild (uses existing kunlun-driver/kunlun.ko).
driver-install-norebuild:
	sudo KL1_P2P_STUB=$${KL1_P2P_STUB:-0} \
		KL1_DMA_DIRECT=$${KL1_DMA_DIRECT:-1} \
		KL1_BOUNCE_PIPE=$${KL1_BOUNCE_PIPE:-1} \
		DISK_ONLY=$${DISK_ONLY:-0} \
		scripts/install_driver.sh

# Install arbitrary .ko to DKMS path only (no rebuild, no unload).
# Example restore stock: make install-ko KO=/path/to/stock.ko
install-ko:
	@test -n "$(KO)" || (echo "usage: make install-ko KO=/path/to/kunlun.ko" >&2; exit 1)
	sudo scripts/install_ko_file.sh "$(KO)"

clean:
	rm -f $(BENCHMARKS) $(TESTS)
	$(MAKE) -C kunlun-driver clean 2>/dev/null || true

run: benchmarks/xpu_perf_test
	./benchmarks/xpu_perf_test 0

# S7 gate: correctness + S4/S5/S6 bandwidth floors
regression: tests/test_p2p_verify tests/test_host_alloc tests/test_pageable_verify \
		tests/test_p2p benchmarks/xpu_perf_test
	scripts/run_driver_regression.sh

# Real-world cases (see docs/plan/20260716-next-phase-plan.md)
cases: benchmarks/xpu_denoise benchmarks/xpu_pipeline_p2p benchmarks/xpu_app_pipeline
	scripts/run_real_cases.sh

# Quick demo path (C2/C3/C4); DEMO_FULL=1 adds C1/C5; DEMO_REGRESSION=1 runs gate
demo:
	scripts/run_demo.sh

.PHONY: all clean run driver driver-install driver-install-norebuild install-ko regression cases demo
