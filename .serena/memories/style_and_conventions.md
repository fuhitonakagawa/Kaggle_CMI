# Code Style and Conventions

## Python Style
- Follow PEP8 guidelines
- Use Google-style docstrings
- All functions must have type hints
- Use descriptive variable names
- Keep functions small and focused (single responsibility)

## Project Structure
```
project_name/
├── config.yaml          # Hyperparameters and settings
├── main.py             # Main execution script
├── src/                # Source code modules
│   ├── __init__.py
│   ├── data_loader.py  # Data loading utilities
│   ├── feature_engineering.py  # Feature extraction
│   ├── model.py        # Model implementation
│   ├── train.py        # Training logic
│   └── evaluate.py     # Evaluation metrics
└── results/            # Output directory
    ├── models/         # Saved models
    ├── logs/           # Training logs
    └── plots/          # Visualizations
```

## Naming Conventions
- Variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase`
- Private methods: `_leading_underscore`
- Files: `snake_case.py`

## Data Science Specific
- Always set random seed for reproducibility
- Use cross-validation for model evaluation
- Log all experiments with timestamp
- Save model checkpoints and metrics
- Use config files for hyperparameters
- Handle missing values explicitly
- Document feature engineering steps

## Git Conventions
- Commit messages with emoji prefixes:
  - 🚀 feat: New feature
  - 🐛 fix: Bug fix
  - 📚 docs: Documentation
  - ♻️ refactor: Code refactoring
  - 🧪 test: Test addition/modification
- Never commit files > 100MB
- Add large files to .gitignore

## Documentation
- Each module should have a module-level docstring
- Each function should have:
  - Brief description
  - Args with types
  - Returns with types
  - Example usage (for complex functions)

## Error Handling
- Use explicit error messages
- Log errors with context
- Fail gracefully with fallback options when possible