#!/bin/bash
set -e  # esce se qualche comando fallisce

# Ottieni la cartella assoluta del progetto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ">>> Project dir: $PROJECT_DIR"

# Cancella il modulo Python compilato
if [ -f "$PROJECT_DIR/exec/main" ]; then
    echo ">>> Removing exec/main"
    rm "$PROJECT_DIR/exec/main"
fi

# Vai nella cartella build e pulisci tutto
BUILD_DIR="$PROJECT_DIR/build"
echo ">>> Cleaning build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
rm -rf *

# CMake + make + make install
echo ">>> Running CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=off

echo ">>> Building..."
make -j$(nproc)


# Torna alla cartella principale
cd "$PROJECT_DIR"
echo ">>> Done! main is now in exec/"