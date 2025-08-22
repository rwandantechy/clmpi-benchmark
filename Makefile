.PHONY: tiny test clean

tiny:
	python scripts/evaluate_models.py --models phi3:mini --label tiny_test

test:
	pytest -q

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +; true
	rm -f results/latest || true
