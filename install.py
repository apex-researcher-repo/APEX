import os

def install_requirements():
    """Install necessary dependencies for the script."""
    requirements = [
        "numpy",
        "pandas",
        "seaborn"
        "matplotlib",
        "scikit-learn",
        "aif360",
        "scipy",
        "torch",
        "botorch",
        "gpytorch",
        "seaborn",
        "scikit-optimize",  # Updated skopt to scikit-optimize
    ]
    
    os.system('python3.11 -m pip install --upgrade pip')
    
    for package in requirements:
        os.system(f'python3.11 -m pip install {package}')

if __name__ == "__main__":
    install_requirements()
