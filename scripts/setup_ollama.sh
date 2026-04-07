#!/bin/bash
# Agent Zero — Ollama setup for Mac Studio M2 Ultra (64GB)
# Run this once after installing Ollama.

set -e

echo "=== Setting Ollama environment variables ==="
launchctl setenv OLLAMA_MAX_LOADED_MODELS "2"
launchctl setenv OLLAMA_NUM_PARALLEL "4"
launchctl setenv OLLAMA_KEEP_ALIVE "1800"
launchctl setenv OLLAMA_FLASH_ATTENTION "1"
launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"
launchctl setenv OLLAMA_HOST "127.0.0.1:11434"

echo ""
echo "Environment variables set. Quit and reopen the Ollama app for them to take effect."
echo ""
read -p "Press Enter after restarting Ollama..."

echo ""
echo "=== Pulling models ==="
echo "Pulling gemma4:31b (~20 GB)..."
ollama pull gemma4:31b

echo "Pulling gemma4:e2b (~2 GB)..."
ollama pull gemma4:e2b

echo ""
echo "=== Pre-warming main model ==="
curl -s http://localhost:11434/api/generate -d '{"model": "gemma4:31b", "prompt": "hello", "stream": false}' > /dev/null

echo ""
echo "=== Done ==="
echo "Verify with: ollama ps"
ollama ps
