"""
Environment setup script for the image captioning system.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_system_dependencies() -> bool:
    """Install system dependencies."""
    system = platform.system().lower()
    
    if system == "linux":
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0 libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev libatlas-base-dev python3-dev"
        ]
    elif system == "darwin":  # macOS
        commands = [
            "brew install libjpeg libpng libtiff libavcodec libavformat libswscale libv4l libxvidcore libx264"
        ]
    elif system == "windows":
        logger.info("Windows detected - please install Visual Studio Build Tools manually")
        return True
    else:
        logger.warning(f"Unknown system: {system}")
        return True
    
    for command in commands:
        if not run_command(command, f"Installing system dependencies"):
            return False
    
    return True


def create_virtual_environment(venv_path: str = "venv") -> bool:
    """Create virtual environment."""
    if Path(venv_path).exists():
        logger.info(f"Virtual environment {venv_path} already exists")
        return True
    
    return run_command(f"python -m venv {venv_path}", "Creating virtual environment")


def activate_virtual_environment(venv_path: str = "venv") -> str:
    """Get activation command for virtual environment."""
    system = platform.system().lower()
    
    if system == "windows":
        return f"{venv_path}\\Scripts\\activate"
    else:
        return f"source {venv_path}/bin/activate"


def install_python_dependencies(venv_path: str = "venv") -> bool:
    """Install Python dependencies."""
    system = platform.system().lower()
    
    if system == "windows":
        pip_path = f"{venv_path}\\Scripts\\pip"
    else:
        pip_path = f"{venv_path}/bin/pip"
    
    commands = [
        f"{pip_path} install --upgrade pip",
        f"{pip_path} install -r requirements.txt"
    ]
    
    for command in commands:
        if not run_command(command, "Installing Python dependencies"):
            return False
    
    return True


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks."""
    commands = [
        "pip install pre-commit",
        "pre-commit install"
    ]
    
    for command in commands:
        if not run_command(command, "Setting up pre-commit hooks"):
            return False
    
    return True


def create_env_file() -> None:
    """Create .env file with default values."""
    env_file = Path(".env")
    
    if env_file.exists():
        logger.info(".env file already exists")
        return
    
    env_content = """# Image Captioning System Environment Variables

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WEB_PORT=8080

# Model Configuration
MODEL_CACHE_DIR=models
DEFAULT_MODEL_TYPE=production
MAX_BATCH_SIZE=32

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Database (if using)
DATABASE_URL=sqlite:///./image_captioning.db

# Redis (if using)
REDIS_URL=redis://localhost:6379

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090

# Safety
ENABLE_NSFW_DETECTION=true
ENABLE_PII_DETECTION=true
ENABLE_CONTENT_MODERATION=true

# Development
DEBUG=false
RELOAD=false
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    logger.info("✓ Created .env file with default values")


def create_gitignore() -> None:
    """Create .gitignore file."""
    gitignore_file = Path(".gitignore")
    
    if gitignore_file.exists():
        logger.info(".gitignore file already exists")
        return
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/
models/
logs/
outputs/
*.pt
*.pth
*.ckpt
.env
.env.local
.env.production

# Jupyter
.ipynb_checkpoints/

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Docker
.dockerignore

# Temporary files
*.tmp
*.temp
"""
    
    with open(gitignore_file, 'w') as f:
        f.write(gitignore_content)
    
    logger.info("✓ Created .gitignore file")


def setup_directories() -> None:
    """Create necessary directories."""
    directories = [
        "data",
        "data/coco",
        "data/conceptual_captions",
        "models",
        "models/clip",
        "models/sample",
        "logs",
        "outputs",
        "configs",
        "tests",
        "docker"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")


def main():
    parser = argparse.ArgumentParser(description="Setup development environment")
    parser.add_argument("--venv", type=str, default="venv", help="Virtual environment path")
    parser.add_argument("--skip-system-deps", action="store_true", help="Skip system dependencies")
    parser.add_argument("--skip-pre-commit", action="store_true", help="Skip pre-commit setup")
    parser.add_argument("--dev", action="store_true", help="Setup for development")
    
    args = parser.parse_args()
    
    logger.info("Setting up Image Captioning System environment...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    if not args.skip_system_deps:
        if not install_system_dependencies():
            logger.warning("System dependencies installation failed, continuing...")
    
    # Create virtual environment
    if not create_virtual_environment(args.venv):
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies(args.venv):
        sys.exit(1)
    
    # Setup pre-commit hooks
    if args.dev and not args.skip_pre_commit:
        if not setup_pre_commit():
            logger.warning("Pre-commit setup failed, continuing...")
    
    # Create configuration files
    create_env_file()
    create_gitignore()
    
    # Setup directories
    setup_directories()
    
    # Print next steps
    logger.info("\n" + "="*50)
    logger.info("SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info(f"To activate the virtual environment, run:")
    logger.info(f"  {activate_virtual_environment(args.venv)}")
    logger.info("\nNext steps:")
    logger.info("1. Activate the virtual environment")
    logger.info("2. Download models and datasets: python scripts/download_models.py --all")
    logger.info("3. Train a model: python train.py --config configs/baseline.yaml --data-config configs/data.yaml")
    logger.info("4. Start the API: python -m src.api.main")
    logger.info("5. Start the web UI: python -m src.web.main")
    logger.info("\nFor Docker deployment:")
    logger.info("  docker-compose -f docker/docker-compose.yml up -d")


if __name__ == "__main__":
    main()
