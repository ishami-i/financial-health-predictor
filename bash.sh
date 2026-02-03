#!/usr/bin/env bash

# Add to your shell profile so DYLD_LIBRARY_PATH persists
echo 'export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc

# Restart Flask
# Press Ctrl+C to stop the current server, then:
source .venv/bin/activate
export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH
python3 api/app.py