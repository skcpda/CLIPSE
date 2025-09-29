.PHONY: setup run-baseline run-debias run-bandpass

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

run-baseline:
	. .venv/bin/activate && python -m src.train --config configs/flickr8k_vitb32_baseline.yaml

run-debias:
	. .venv/bin/activate && python -m src.train --config configs/flickr8k_vitb32_debias.yaml

run-bandpass:
	. .venv/bin/activate && python -m src.train --config configs/flickr8k_vitb32_bandpass.yaml
