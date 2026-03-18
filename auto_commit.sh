#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./auto_commit.sh [options]

Options:
  -m, --message <msg>   Commit message (default: timestamped auto-commit message)
  -r, --remote <name>   Remote to push to (default: origin)
  -b, --branch <name>   Branch to push to (default: current branch)
  -h, --help            Show this help
EOF
}

commit_message=""
remote="origin"
branch=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      shift
      [[ $# -gt 0 ]] || { echo "Error: missing value for --message"; exit 1; }
      commit_message="$1"
      ;;
    -r|--remote)
      shift
      [[ $# -gt 0 ]] || { echo "Error: missing value for --remote"; exit 1; }
      remote="$1"
      ;;
    -b|--branch)
      shift
      [[ $# -gt 0 ]] || { echo "Error: missing value for --branch"; exit 1; }
      branch="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'"
      usage
      exit 1
      ;;
  esac
  shift
done

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Error: this is not a git repository."
  exit 1
}

if [[ -z "$branch" ]]; then
  branch="$(git rev-parse --abbrev-ref HEAD)"
fi

if [[ -z "$commit_message" ]]; then
  commit_message="chore: auto-commit $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
fi

git add -A

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$commit_message"

if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git push
else
  git push -u "$remote" "$branch"
fi

echo "Done: committed and pushed to ${remote}/${branch}"
