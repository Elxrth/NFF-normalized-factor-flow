$ErrorActionPreference = "Stop"

$Repo = "https://github.com/Elxrth/NFF-normalized-factor-flow.git"
$InstallPath = "$env:USERPROFILE\NFF"

Write-Host "Installing NFF..."

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is required but was not found."
    Write-Host "Install Python 3.10+ and run the installer again."
    exit 1
}

# Check Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is required but was not found."
    Write-Host "Install Git and run the installer again."
    exit 1
}

# Clone repository
if (-not (Test-Path $InstallPath)) {
    git clone $Repo $InstallPath
}
else {
    Write-Host "NFF already exists at $InstallPath"
}

Set-Location $InstallPath

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
& ".\.venv\Scripts\Activate.ps1"

# Update pip
python -m pip install --upgrade pip

# Install PyTorch
Write-Host "Installing PyTorch..."
python -m pip install torch

Write-Host ""
Write-Host "NFF installed successfully."
Write-Host "Location: $InstallPath"
Write-Host ""
Write-Host "Starting NFF..."

python main.py
