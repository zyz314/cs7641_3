# Installation

Install Poetry and add it to `PATH`:

```
curl -sSL https://install.python-poetry.org | python -
echo 'export PATH="/home/$USER/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Test installation of poetry
poetry --version
```

Install project dependencies and build the project:
```
poetry install
poetry build
```
