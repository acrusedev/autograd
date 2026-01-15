# autograd

Library with a Rust backend and Python frontend for simple neural networks.

## Prerequisites

- **Rust** (1.70+): [Install via rustup](https://rustup.rs/)
- **Python** (3.8+): [Download Python](https://www.python.org/downloads/)
- **maturin**: Python package for building Rust extensions

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/autograd.git
cd autograd
```

### 2. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
```

### 3. Install maturin

```bash
pip install maturin
```

### 4. Build and install

```bash
maturin develop
```

This compiles the Rust code and installs the package into your virtual environment under the name of autograd_core.

### 5. Test the installation

```python
# Root directory
PYTHONPATH=. python -c "from autograd import Tensor; t = Tensor([1,2,3,4]); print(t.shape)" # output: (4,)
```
