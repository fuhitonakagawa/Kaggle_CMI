# Suggested Commands for Development

## Package Management (using uv only)
- Install dependencies: `uv sync`
- Add a package: `uv add package_name`
- Add dev dependency: `uv add --dev package_name`
- Run a command: `uv run python main.py`

## Training and Evaluation
- Run baseline training: `cd 1_IMU_baseline_lgbm_20250112 && uv run python main.py`
- Run specific notebook: `uv run jupyter notebook notebooks-IMU/notebook_name.ipynb`

## Data Exploration
- List data files: `ls cmi-detect-behavior-with-sensor-data/`
- Check data shape: `uv run python -c "import pandas as pd; df = pd.read_csv('cmi-detect-behavior-with-sensor-data/train.csv'); print(df.shape)"`

## Git Commands
- Check status: `git status`
- Add files: `git add .`
- Commit: `git commit -m "message"`
- View log: `git log --oneline`

## System Commands (macOS)
- List files: `ls -la`
- Navigate: `cd directory`
- Create directory: `mkdir -p path/to/dir`
- Find files: `find . -name "*.py"`
- Search in files: `grep -r "pattern" .`

## Testing (when implemented)
- Run tests: `uv run pytest tests/`
- Run with coverage: `uv run pytest --cov=src tests/`