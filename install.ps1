$ErrorActionPreference = "Stop"

$Repo = "https://github.com/Elxrth/NFF-normalized-factor-flow/archive/refs/heads/main.zip"
$InstallPath = "$env:USERPROFILE\NFF"

Write-Host "Installing NFF..."

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python was not found."
    Write-Host "Please install Python 3.10 or newer."
    exit 1
}

# Download repository
$TempZip = "$env:TEMP\nff.zip"
$TempDir = "$env:TEMP\nff-install"

if (Test-Path $TempDir) {
    Remove-Item $TempDir -Recurse -Force
}

Write-Host "Downloading NFF..."
Invoke-WebRequest -Uri $Repo -OutFile $TempZip

# Extract
Write-Host "Extracting files..."
Expand-Archive -Path $TempZip -DestinationPath $TempDir -Force

$SourcePath = Join-Path $TempDir "NFF-normalized-factor-flow-main"

# Install directory
if (-not (Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath | Out-Null
}

Copy-Item "$SourcePath\*" $InstallPath -Recurse -Force

Set-Location $InstallPath

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

$Python = ".\.venv\Scripts\python.exe"

# Upgrade pip
Write-Host "Updating pip..."
& $Python -m pip install --upgrade pip

# Install PyTorch
Write-Host "Installing PyTorch..."
& $Python -m pip install torch

Write-Host ""
Write-Host "NFF installed successfully."
Write-Host "Location: $InstallPath"
Write-Host ""

# Run NFF
Write-Host "Starting NFF..."
& $Python main.py
