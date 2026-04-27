.PHONY: install

install:
	uv venv --python 3.12 --python-preference only-managed
	cd deps/vllm && git checkout releases/v0.15.0 && VLLM_USE_PRECOMPILED=1 uv pip install -e . && git checkout bs/lcspec
	cd deps/speculators && git checkout v0.4.0 && uv pip install -e . && git checkout bs/lcspec
	uv pip install -e .
