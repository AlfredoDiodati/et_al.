#!/bin/bash

# Install git hooks for this repository.
# Run once after cloning: bash scripts/install-hooks.sh

cd "$(dirname "$0")/.."

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
./check.sh
EOF
chmod +x .git/hooks/pre-commit
echo "pre-commit hook installed"
