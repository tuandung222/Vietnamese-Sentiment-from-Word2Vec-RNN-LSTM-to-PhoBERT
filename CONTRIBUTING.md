# Contributing to Vietnamese Sentiment Analysis

Thank you for your interest in contributing to Vietnamese Sentiment Analysis! This document provides guidelines for contributing to this project.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker
- Include detailed steps to reproduce the bug
- Include system information (OS, Python version, etc.)
- Include error messages and stack traces

### Suggesting Enhancements

- Use the GitHub issue tracker
- Describe the enhancement in detail
- Explain why this enhancement would be useful
- Include any mockups or examples if applicable

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Add unit tests for new features
- Use Black for code formatting
- Use isort for import sorting

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Use descriptive test names

## Documentation

- Update README.md if needed
- Add docstrings to new functions
- Update API documentation
- Include usage examples

## Commit Messages

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers.