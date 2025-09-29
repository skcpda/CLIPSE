.PHONY: setup run-baseline run-debias run-bandpass run-all-seeds

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

run-baseline:
	. .venv/bin/activate && python -m src.train_advanced --config configs/flickr8k_vitb32_baseline.yaml

run-debias:
	. .venv/bin/activate && python -m src.train_advanced --config configs/flickr8k_vitb32_debias.yaml

run-bandpass:
	. .venv/bin/activate && python -m src.train_advanced --config configs/flickr8k_vitb32_bandpass.yaml

# Run all three experiments with all three seeds
run-all-seeds:
	@echo "Running baseline experiments..."
	@for seed in 13 17 23; do \
		sed "s/seed: 13/seed: $$seed/g" configs/flickr8k_vitb32_baseline.yaml > configs/flickr8k_baseline_seed$$seed.yaml; \
		. .venv/bin/activate && python -m src.train_advanced --config configs/flickr8k_baseline_seed$$seed.yaml; \
	done
	@echo "Running SANW debias experiments..."
	@for seed in 13 17 23; do \
		sed "s/seed: 13/seed: $$seed/g" configs/flickr8k_vitb32_debias.yaml > configs/flickr8k_debias_seed$$seed.yaml; \
		. .venv/bin/activate && python -m src.train_advanced --config configs/flickr8k_debias_seed$$seed.yaml; \
	done
	@echo "Running SANW bandpass experiments..."
	@for seed in 13 17 23; do \
		sed "s/seed: 13/seed: $$seed/g" configs/flickr8k_vitb32_bandpass.yaml > configs/flickr8k_bandpass_seed$$seed.yaml; \
		. .venv/bin/activate && python -m src.train_advanced --config configs/flickr8k_bandpass_seed$$seed.yaml; \
	done
