# Qwen pre-flip confidence

Reproduction package for the causal confidence experiment.

## Analysis

```bash
pip install -r requirements.txt
python -m src.analyze
pytest -q
```

## Model runs

Use a local copy of the official Qwen/Qwen3-0.6B checkpoint.

```bash
python src/validate_choice.py --model /path/to/Qwen3-0.6B --index 0
python src/run_orientation.py --model /path/to/Qwen3-0.6B --split test --index 0 --orientation original
python src/run_orientation.py --model /path/to/Qwen3-0.6B --split test --index 0 --orientation semantic_swapped
```

Repeat validation for indices 0 through 3 and both test orientations for indices 0 through 3.

