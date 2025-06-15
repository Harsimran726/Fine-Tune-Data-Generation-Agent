#!/usr/bin/env bash
echo "🚀 Starting custom build..."

# Upgrade pip
pip install --upgrade pip

# Install all packages WITHOUT isolated build
pip install --no-cache-dir --no-build-isolation -r requirements.txt

echo "✅ Build completed successfully!"
