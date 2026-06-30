#!/bin/bash

# Install git hooks for this repository.
# Run once after cloning: bash scripts/install-hooks.sh

cd "$(dirname "$0")/.."

cp check.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "pre-commit hook installed"
