#!/bin/bash

echo "Checking required dependencies..."
echo "================================"

missing_deps=()

# Check for uv
if command -v uv &> /dev/null; then
    echo "✓ uv is installed ($(uv --version))"
else
    echo "✗ uv is NOT installed"
    missing_deps+=("uv")
fi

# Check for git-lfs
if command -v git-lfs &> /dev/null; then
    echo "✓ git-lfs is installed ($(git-lfs --version))"
else
    echo "✗ git-lfs is NOT installed"
    missing_deps+=("git-lfs")
fi

# Check for ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg is installed ($(ffmpeg -version | head -n1))"
else
    echo "✗ ffmpeg is NOT installed"
    missing_deps+=("ffmpeg")
fi

echo "================================"

# Summary
if [ ${#missing_deps[@]} -eq 0 ]; then
    echo "All system dependencies are installed!"
    exit 0
else
    echo "Missing dependencies: ${missing_deps[*]}"
    echo "Please install the missing tools and try again."
    exit 1
fi
