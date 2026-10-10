# Run pntx jobs on the home GPU server: sync code, run remotely, fetch results.
# Adapted from github.com/pillyshi/gpu-makefiles (home/ template + common.mk),
# without the run-metadata step.
#
# Configure in .env (git-ignored), or pass on the command line:
#   HOST=<ssh host or alias>   REMOTE_DIR=<dir under remote $HOME, default: pntx>
#
# Remote prerequisites: uv, rsync, tmux, git; CUDA toolkit + gcc for the
# llama-cpp-python CUDA build (see install-llama-cpp).
#
# Note on uv: llama-cpp-python is built with CUDA via `uv pip install`, which
# a plain `uv sync` would uninstall again (sync removes packages that don't
# match the lockfile selection). So every remote command here uses
# `uv sync --inexact` / `uv run --no-sync`.

-include .env

REMOTE_DIR ?= $(shell basename $(CURDIR))

CUDA_VERSION      ?= 12.8
GCC_VERSION       ?= 13
LLAMA_CPP_VERSION ?= 0.3.34  # keep in step with uv.lock

# Benchmark defaults (override per call).
SEED        ?= 0
N_GPU_LAYERS ?= -1
FT_MODEL    ?= bert-base-uncased

RESULTS := benchmarks/results

define remote-exec
ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && $(1)'"
endef

# Start a command in a detached tmux session (survives ssh disconnects); its
# output is also written to $(RESULTS)/<session>.log, which `make fetch` brings back.
# For the fixed internal commands below (no quotes inside).
define remote-exec-tmux
ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && mkdir -p $(RESULTS) && tmux new-session -d -s $(1) \"$(2) 2>&1 | tee $(RESULTS)/$(1).log\"'"
endef

# User-supplied CMD is sent as a script on stdin (or as a script file for
# tmux) instead of being nested inside bash -c '...', so its quotes are passed
# through untouched. CMD reaches the recipe shell via the environment, taken
# with $(value ...) so make doesn't expand $-references in it ($s, $HOME, ...).
export RUN_CMD := $(value CMD)

.PHONY: help sync fetch run run-bg ssh attach status install install-llama-cpp setup \
	test-integration bench-generate bench-eval-ft guard-%

help:
	@echo "Targets (HOST from .env or HOST=...):"
	@echo "  setup                 uv sync (finetuning extra + benchmark group) and CUDA llama-cpp-python"
	@echo "  sync / fetch          push code (+ local benchmark results) / pull $(RESULTS)/"
	@echo "  run CMD='...'         sync, run CMD remotely (foreground), fetch"
	@echo "  run-bg SESSION=x CMD='...'  sync, run CMD in tmux session x; later: attach / fetch"
	@echo "  status                tmux sessions and GPU usage"
	@echo "  test-integration MODEL_PATH=/remote/model.gguf"
	@echo "  bench-generate MODEL_PATH=/remote/model.gguf [SEED=0]   (tmux session gen-seed<SEED>)"
	@echo "  bench-eval-ft AUG=$(RESULTS)/<aug>.json [FT_MODEL=bert-base-uncased]"
	@echo "CMD is sent to the remote shell as-is (any quoting your local shell passes through)."

guard-%:
	$(if $(value $*),,$(error $* is not set. Define it in .env or pass $*=...))

# --- transfer -------------------------------------------------------------------

# .gitignore files (including nested ones) are honoured, except that local
# benchmark results are pushed too, so augmentations generated locally can be
# evaluated remotely. -u never overwrites a newer file on the receiver.
sync: guard-HOST
	rsync -avzu --exclude='.git' \
		--include='$(RESULTS)/' --include='$(RESULTS)/***' \
		--filter=':- .gitignore' \
		./ $(HOST):$(REMOTE_DIR)/

fetch: guard-HOST
	rsync -avzu --include='benchmarks/' --include='$(RESULTS)/***' --exclude='*' \
		$(HOST):$(REMOTE_DIR)/ ./

# --- running --------------------------------------------------------------------

run: sync guard-CMD
	printf '%s\n' "cd $(REMOTE_DIR)" "$$RUN_CMD" | ssh $(HOST) 'bash -l -s'
	$(MAKE) fetch

run-bg: sync guard-CMD guard-SESSION
	printf '%s\n' "cd ~/$(REMOTE_DIR)" "$$RUN_CMD" | ssh $(HOST) 'cat > ~/$(REMOTE_DIR)/.run-$(SESSION).sh'
	$(call remote-exec-tmux,$(SESSION),bash -l .run-$(SESSION).sh)
	@echo "Started tmux session '$(SESSION)'. Follow with: make attach SESSION=$(SESSION); collect with: make fetch"

ssh: guard-HOST
	ssh $(HOST)

attach: guard-HOST guard-SESSION
	ssh -t $(HOST) "tmux attach -t $(SESSION)"

status: guard-HOST
	-ssh $(HOST) "tmux ls"
	ssh $(HOST) "bash -l -c 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv'"

# --- environment ----------------------------------------------------------------

install: sync
	$(call remote-exec,uv sync --inexact --extra finetuning --group benchmark)

install-llama-cpp: guard-HOST
	ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && \
		CUDA_HOME=/usr/local/cuda-$(CUDA_VERSION) \
		PATH=/usr/local/cuda-$(CUDA_VERSION)/bin:\$$HOME/.local/bin:\$$PATH \
		CMAKE_ARGS=\"-DGGML_CUDA=on -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-$(GCC_VERSION)\" \
		CC=/usr/bin/gcc-$(GCC_VERSION) CXX=/usr/bin/g++-$(GCC_VERSION) \
		uv pip install \"llama-cpp-python==$(strip $(LLAMA_CPP_VERSION))\" --no-binary llama-cpp-python'"

setup: install install-llama-cpp

# --- pntx jobs ------------------------------------------------------------------

test-integration: sync guard-MODEL_PATH
	$(call remote-exec,PNTX_LLAMA_MODEL_PATH=$(MODEL_PATH) uv run --no-sync pytest tests/integration -q)

# Generation takes hours per seed, so it runs in tmux; `make fetch` when done.
bench-generate: sync guard-MODEL_PATH
	$(call remote-exec-tmux,gen-seed$(SEED),uv run --no-sync python -m benchmarks.pn2t.downstream --seed $(SEED) --output $(RESULTS)/pn2t-downstream-aug-seed$(SEED).json generate --editor-model $(MODEL_PATH) --n-gpu-layers $(N_GPU_LAYERS))
	@echo "Started tmux session 'gen-seed$(SEED)'."

bench-eval-ft: sync guard-AUG
	$(call remote-exec,uv run --no-sync python -m benchmarks.pn2t.downstream evaluate --augmentations $(AUG) --classifier finetuning --model-name $(FT_MODEL))
	$(MAKE) fetch
