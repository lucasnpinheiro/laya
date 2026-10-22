#!/bin/sh
# Ativa os git hooks versionados em .githooks/ neste clone.
set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/commit-msg

echo "Hooks ativos: core.hooksPath=.githooks"
if [ ! -f .githooks/denylist.local ]; then
  echo "Crie a denylist local (não versionada):"
  echo "  cp .githooks/denylist.local.example .githooks/denylist.local"
  echo "  e acrescente os termos sensíveis (nomes de clientes, projetos internos)."
fi
