# Contributing to AbstractIntegratedModule

First off, thank you for considering contributing to AbstractIntegratedModule!

## 📋 Table of Contents
- [How Can I Contribute?](#how-can-i-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Pull Requests](#pull-requests)


## How Can I Contribute?

### 🐛 Reporting Bugs

**Before submitting a bug report:**
- Check the [Issues](https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework/issues) to see if it's already reported
- Test with the latest version

**How to submit a bug report:**
1. Go to [Issues](https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework/issues)
2. Click "New Issue"
3. Select "Bug Report" template
4. Fill in:
   - **Title**: Clear, descriptive summary
   - **Environment**: OS, Python version, CPU architecture
   - **Steps to Reproduce**: Minimal code example
   - **Expected behavior**: What should happen
   - **Actual behavior**: What happened instead
   - **Screenshots/Logs**: If applicable

### 💡 Suggesting Enhancements

**Before submitting:**
- Check existing issues for similar suggestions
- Consider if it fits the project's scope

**How to submit:**
1. Go to [Issues](https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework/issues)
2. Click "New Issue"
3. Select "Feature Request" template
4. Describe:
   - **Use case**: What problem does it solve?
   - **Proposed solution**: How would it work?
   - **Alternatives**: Other approaches considered

### 🔧 Pull Requests

**Step-by-step:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

**PR Requirements:**
- One clear purpose per PR
- Update documentation if needed
- Add tests for new features
- Keep changes focused (avoid unrelated fixes)
## 💻 Development Setup

### Prerequisites
```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements-dev.txt
```

### Building from Source

```bash
# Clone your fork
git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Running the Test Suite

```bash
# All tests
pytest

# Specific test
pytest tests/main.py 

# With coverage
pytest --cov=main tests/
```

### 📐 Style Guidelines

Python Style

· Follow PEP 8
· Use type hints where possible
· Maximum line length: 88 characters (Black default)

```python
# Good
def predict(text: str, confidence_threshold: float = 0.5) -> dict:
    """Predict activity from window title."""
    pass

# Avoid
def predict(text,thresh=0.5):
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
```


### 🤝 Community

Contact

· GitHub Issues: For bugs and feature requests
· GitHub Discussions: For questions and ideas
· Email: hernikpuspita5@gmail.com (for security issues or other necessary things only)

### Recognition

Contributors will be added to README.md.

### 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

