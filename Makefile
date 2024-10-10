run-one-s2:
	python src/run_task.py \
		--tile-id=68,18 \
		--datetime=2024 \
		--version=0.0.1 \
		--output-bucket=dep-public-staging \
		--base-product="s2" \
		--decimated \
		--no-all-bands \
		--memory-limit 10GB \
		--overwrite
