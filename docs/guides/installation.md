# Installation Guide

Detailed installation instructions for the Titanic Survival Prediction system.

---

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.11 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Disk Space**: 2GB for dependencies and models
- **Internet**: Required for initial setup

### Recommended Requirements

- **RAM**: 4GB or more
- **CPU**: 4+ cores for faster training
- **SSD**: For better I/O performance

---

## Installation Methods

Choose the method that best fits your needs:

=== "uv (Recommended)"

    Fastest method using the modern uv package manager:

    ```bash
    # Install uv if not already installed
    curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

    # Clone and install
    git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
    cd Titanic-Machine-Learning-from-Disaster
    uv sync

    # Run
    python app.py
    ```

=== "pip"

    Traditional method using pip:

    ```bash
    # Clone repository
    git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
    cd Titanic-Machine-Learning-from-Disaster

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    .venv\Scripts\activate  # Windows

    # Install dependencies
    pip install -r requirements.txt

    # Run
    python app.py
    ```

=== "Docker"

    No Python installation needed:

    ```bash
    # Clone repository
    git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
    cd Titanic-Machine-Learning-from-Disaster

    # Build and run
    docker-compose up
    ```

    Access at http://localhost:5000

---

## Detailed Setup

### 1. Install Python

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- Check "Add Python to PATH" during installation
- Verify: `python --version`

**macOS:**
    ```bash
    # Using Homebrew
    brew install python@3.11
    ```

**Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3-pip
    ```

### 2. Install Git

**Windows:**
- Download from [git-scm.com](https://git-scm.com/download/win)

**Linux:**
    ```bash
    sudo apt install git
    ```

### 3. Clone Repository

    ```bash
    git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
    cd Titanic-Machine-Learning-from-Disaster
    ```

### 4. Install Dependencies

#### Option A: Using uv (Fastest)

    ```bash
    # Install uv
    pip install uv

    # Install all dependencies
    uv sync

    # Install with optional groups
    uv sync --group dev        # Development tools
    uv sync --group notebooks  # Jupyter support
    uv sync --group docs       # Documentation tools
    uv sync --all-groups       # Everything
    ```

#### Option B: Using pip

    ```bash
    # Create virtual environment
    python -m venv .venv

    # Activate virtual environment
    # Windows PowerShell:
    .venv\Scripts\Activate.ps1
    # Windows CMD:
    .venv\Scripts\activate.bat
    # macOS/Linux:
    source .venv/bin/activate

    # Upgrade pip
    python -m pip install --upgrade pip

    # Install dependencies
    pip install -r requirements.txt

    # Optional: Install development dependencies
    pip install pytest pytest-cov black flake8 mypy
    ```

### 5. Verify Installation

    ```bash
    # Check Python packages
    python -c "import pandas, numpy, sklearn, xgboost, catboost, flask; print('✓ All core packages installed')"

    # Run tests
    pytest tests/

    # Check Flask app
    python app.py &
    curl http://localhost:5000/health
    ```

---

## Post-Installation Setup

### Download Pre-trained Models (Optional)

If models aren't included in the repository:

    ```bash
    # Download from releases or train your own
    python scripts/run_training.py
    ```

Models will be saved to `models/` directory.

### Configure Environment (Optional)

Create `.env` file for custom settings:

    ```bash
    # .env
    FLASK_ENV=development
    FLASK_DEBUG=True
    LOG_LEVEL=DEBUG
    MODEL_PATH=models/custom_model.pkl
    ```

---

## Development Setup

For contributors and developers:

### 1. Install Development Dependencies

    ```bash
    uv sync --all-groups
    ```

### 2. Install Pre-commit Hooks

    ```bash
    pip install pre-commit
    pre-commit install
    ```

### 3. Configure IDE

**VS Code** (`settings.json`):
    ```json
    {
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
    }
    ```

**PyCharm**:
- File → Settings → Tools → Python Integrated Tools
- Set Default test runner to pytest
- Set Docstring format to Google

---

## Troubleshooting

### Common Issues

#### 1. `uv` command not found

    ```bash
    # Reinstall uv
    pip install --user --upgrade uv

    # Or use full path
    python -m uv sync
    ```

#### 2. Permission denied errors

**Windows**: Run as Administrator  
**macOS/Linux**: Use `sudo` or fix permissions:

    ```bash
    sudo chown -R $USER:$USER .
    ```

#### 3. SSL Certificate errors

    ```bash
    # Upgrade certificates
    pip install --upgrade certifi
    ```

#### 4. Module not found errors

    ```bash
    # Ensure virtual environment is activated
    which python  # Should point to .venv/bin/python

    # Reinstall dependencies
    pip install --force-reinstall -r requirements.txt
    ```

#### 5. Port 5000 already in use

    ```bash
    # Find process using port
    # Windows:
    netstat -ano | findstr :5000
    # macOS/Linux:
    lsof -i :5000

    # Kill process or use different port
    # Edit app.py: app.run(port=8000)
    ```

---

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt, not Git Bash
- Some packages require Microsoft C++ Build Tools
- Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### macOS (Apple Silicon)

    ```bash
    # Use Rosetta for compatibility
    arch -x86_64 python -m pip install -r requirements.txt
    ```

### Linux

- Install system dependencies for NumPy/SciPy:

        ```bash
        sudo apt install python3-dev build-essential libatlas-base-dev
        ```

    ---

## Uninstallation

    ```bash
    # Remove virtual environment
    rm -rf .venv  # Unix
    rmdir /s .venv  # Windows

    # Remove repository
    cd ..
    rm -rf Titanic-Machine-Learning-from-Disaster
    ```

---

## Next Steps

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Run your first prediction

    [:octicons-arrow-right-24: Quick Start Guide](quickstart.md)

- :material-book-open-variant:{ .lg .middle } **Explore Notebook**

    ---

    Dive into the analysis

    [:octicons-arrow-right-24: Notebook Guide](notebook-guide.md)

:material-code-braces:{ .lg .middle } **API Reference**

    ---

    Integrate into your app

    [:octicons-arrow-right-24: API Docs](../reference/api.md)

</div>

---

## Need Help?

- Check [FAQ](faq.md)
- Review [Troubleshooting](#troubleshooting)
- Open an issue on [GitHub](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/issues)
- Email: k.s.bonev@gmail.com
