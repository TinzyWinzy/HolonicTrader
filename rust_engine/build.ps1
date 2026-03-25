# Build script for holonic_speed with Python 3.13
$env:PYO3_PYTHON = "c:\Users\USER\Documents\AEHML\HolonicTrader\.venv313\Scripts\python.exe"
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"
Write-Host "Building with Python: $env:PYO3_PYTHON"
Write-Host "Features: $env:CARGO_FEATURES"

cargo clean

if ($env:CARGO_FEATURES -eq "no-onnx") {
    Write-Host "Building WITHOUT ONNX support..."
    cargo build --release --no-default-features
} else {
    Write-Host "Building WITH ONNX support..."
    cargo build --release --features onnx
}

Write-Host "Build complete. Copying .pyd to HolonicTrader directory..."
Copy-Item "target\release\holonic_speed.dll" -Destination "..\holonic_speed.pyd" -Force
Write-Host "Done!"
