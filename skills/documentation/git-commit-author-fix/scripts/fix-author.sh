#!/usr/bin/env bash
# fix-author.sh — 修复 GitCode MR 单次提交的作者信息
# Usage: ./fix-author.sh <fork-url> <branch> <name> <email>

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <fork-url> <branch> <name> <email>"
    echo "Example: $0 https://gitcode.com/user/repo.git feat/my-branch 'Correct Name' correct@email.com"
    exit 1
fi

FORK_URL="$1"
BRANCH="$2"
AUTHOR_NAME="$3"
AUTHOR_EMAIL="$4"

echo "=== Step 1: Fetch fork branch ==="
git remote remove fork 2>/dev/null || true
git remote add fork "$FORK_URL"
git fetch fork "$BRANCH"

echo "=== Step 2: Checkout branch ==="
git checkout -b "$BRANCH" "fork/$BRANCH"

echo "=== Step 3: Fix author ==="
git commit --amend --author="$AUTHOR_NAME <$AUTHOR_EMAIL>" --no-edit

echo "=== Step 4: Verify ==="
git log --format="%h %an <%ae> %s" -1
