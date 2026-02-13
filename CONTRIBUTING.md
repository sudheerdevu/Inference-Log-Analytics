# Contributing to Inference Log Analytics

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone and setup
git clone https://github.com/sudheerdevu/Inference-Log-Analytics.git
cd Inference-Log-Analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Code Style

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
```

## Adding New Parsers

1. Create new parser in `src/parsers/`
2. Inherit from `BaseParser`
3. Implement `parse()` and `extract_metrics()`
4. Add tests
5. Update documentation

## Pull Request Process

1. Fork the repository
2. Create feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit PR

## License

Contributions are licensed under MIT License.
