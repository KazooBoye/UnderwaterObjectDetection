#!/usr/bin/env python3
"""
Single Configuration and Setup Script
Handles both configuration initialization and environment setup
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def initialize_config():
    """Initialize configuration files for the project"""
    
    project_root = Path(__file__).parent.absolute()
    
    print("=== Initializing Configuration Files ===")
    
    # 1. Create config.py if it doesn't exist
    config_file = project_root / "config.py"
    config_template = project_root / "config_template.py"
    
    if not config_file.exists():
        print("Creating config.py from template...")
        shutil.copy(config_template, config_file)
        print("Created config.py")
    else:
        print("config.py already exists")
    
    # 2. Create/update data.yaml with dynamic path
    data_yaml_path = project_root / "aquarium_pretrain" / "data.yaml"
    data_yaml_content = f"""path: {project_root / "aquarium_pretrain"}
train: train/images
val: valid/images
test: test/images

nc: 7
names: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

# Class weights based on analysis (inverse frequency)
# fish: 1.0, jellyfish: 5.1, penguin: 6.3, puffin: 11.2, shark: 3.6, starfish: 25.1, stingray: 14.4
"""
    
    print("Updating aquarium_pretrain/data.yaml with current path...")
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    print("Updated data.yaml")
    
    # 3. Import and test config
    try:
        from config import Config
        print("Configuration loaded successfully")
        print(f"   Project Root: {Config.get_project_root()}")
        print(f"   Dataset Path: {Config.get_data_yaml_path()}")
        print(f"   Runs Directory: {Config.get_runs_dir()}")
        return True
    except ImportError as e:
        print(f"Error importing config: {e}")
        return False

def setup_environment():
    """Setup the training environment based on available options"""
    
    print("\n=== Environment Setup ===")
    print("Choose your setup method:")
    print("1. Anaconda/Miniconda (recommended)")
    print("2. Python venv (fallback)")
    print("3. Skip environment setup")
    
    choice = input("Enter choice [1/2/3]: ").strip()
    
    if choice == "1":
        return setup_conda()
    elif choice == "2":
        return setup_venv()
    elif choice == "3":
        print("Skipping environment setup. Make sure you have the required packages installed.")
        return True
    else:
        print("Invalid choice.")
        return False

def setup_conda():
    """Setup Conda environment"""
    
    # Check if conda is available
    if not shutil.which("conda"):
        print("Conda not found.")
        install_conda = input("Install Miniconda automatically? [y/N]: ").lower() == 'y'
        
        if install_conda:
            print("Installing Miniconda...")
            try:
                os.chdir("/tmp")
                subprocess.run(["wget", "-q", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh", "-O", "miniconda.sh"], check=True)
                subprocess.run(["bash", "miniconda.sh", "-b", "-p", f"{os.path.expanduser('~')}/miniconda3"], check=True)
                
                print("Miniconda installed. Please restart your terminal and run this script again.")
                print("Or run: source ~/.bashrc && python3 init_config.py")
                return True
            except subprocess.CalledProcessError:
                print("Failed to install Miniconda. Please install manually.")
                return False
        else:
            return setup_venv()
    
    print("Conda found")
    
    # Environment setup
    env_name = "yolov8_aquarium"
    
    try:
        # Create environment
        print(f"Creating conda environment: {env_name}")
        subprocess.run(["conda", "create", "-n", env_name, "python=3.9", "-y"], check=True, capture_output=True)
        
        # Install packages using environment.yml if it exists
        env_file = Path("environment.yml")
        if env_file.exists():
            print("Installing packages from environment.yml...")
            subprocess.run(["conda", "env", "update", "-n", env_name, "-f", str(env_file)], check=True)
        else:
            # Manual package installation
            print("Installing core packages...")
            
            # Check for GPU and install appropriate PyTorch
            if shutil.which("nvidia-smi"):
                print("NVIDIA GPU detected, installing PyTorch with CUDA...")
                subprocess.run(["conda", "install", "-n", env_name, "pytorch", "torchvision", "torchaudio", "pytorch-cuda=11.8", "-c", "pytorch", "-c", "nvidia", "-y"], check=True)
            else:
                print("No NVIDIA GPU detected, installing CPU-only PyTorch...")
                subprocess.run(["conda", "install", "-n", env_name, "pytorch", "torchvision", "torchaudio", "cpuonly", "-c", "pytorch", "-y"], check=True)
            
            # Install other packages
            subprocess.run(["conda", "install", "-n", env_name, "-c", "conda-forge", "numpy", "opencv", "matplotlib", "seaborn", "pandas", "pyyaml", "tqdm", "requests", "pillow", "jupyter", "ipywidgets", "-y"], check=True)
            
            # Install ultralytics via pip in the conda environment  
            subprocess.run(["conda", "run", "-n", env_name, "pip", "install", "ultralytics>=8.0.0"], check=True)
        
        print(f"Conda environment '{env_name}' created successfully!")
        print(f"To activate: conda activate {env_name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup conda environment: {e}")
        return setup_venv()

def setup_venv():
    """Setup Python virtual environment"""
    
    # Check Python 3
    if not shutil.which("python3"):
        print("Python 3 not found. Please install Python 3.8+")
        return False
    
    print("Python 3 found")
    
    env_dir = "venv_yolov8"
    
    try:
        # Create virtual environment
        print(f"Creating Python virtual environment: {env_dir}")
        subprocess.run([sys.executable, "-m", "venv", env_dir], check=True)
        
        # Get activation script path
        if os.name == 'nt':  # Windows
            activate_script = os.path.join(env_dir, "Scripts", "activate")
        else:  # Unix-like
            activate_script = os.path.join(env_dir, "bin", "activate")
        
        # Install packages
        pip_exe = os.path.join(env_dir, "bin", "python") if os.name != 'nt' else os.path.join(env_dir, "Scripts", "python.exe")
        
        print("Upgrading pip...")
        subprocess.run([pip_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        print("Installing PyTorch (CPU version)...")
        subprocess.run([pip_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
        
        print("Installing other packages...")
        subprocess.run([pip_exe, "-m", "pip", "install", "ultralytics>=8.0.0", "numpy", "opencv-python", "pillow", "matplotlib", "seaborn", "pandas", "pyyaml", "tqdm", "requests", "jupyter", "ipywidgets"], check=True)
        
        print(f"Virtual environment '{env_dir}' created successfully!")
        print(f"To activate: source {activate_script}")
        
        # Create activation helper
        with open("activate_env.sh", "w") as f:
            f.write(f"#!/bin/bash\nsource {activate_script}\necho 'YOLOv8 environment activated!'\n")
        os.chmod("activate_env.sh", 0o755)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup virtual environment: {e}")
        return False

def main():
    """Main setup function"""
    print("=== YOLOv8 Underwater Object Detection Setup ===")
    print("")
    
    # Initialize configuration
    if not initialize_config():
        print("Configuration initialization failed")
        return False
    
    # Setup environment
    if not setup_environment():
        print("Environment setup failed")
        return False
    
    print("")
    print("=== Setup Complete ===")
    print("Your YOLOv8 underwater object detection project is ready!")
    print("")
    print("Next steps:")
    print("1. Activate your environment (conda activate yolov8_aquarium OR source venv_yolov8/bin/activate)")
    print("2. Run preprocessing: python3 preprocess_data.py")
    print("3. Start training: python3 train_yolov8.py")
    print("")
    print("See README.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
