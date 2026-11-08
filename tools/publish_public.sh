#!/usr/bin/env bash
# Publish the current commit of this (private) repository to the public copy,
# github.com/hkm4lmn/SCI-RTW-public, as one new commit.
#
#   bash tools/publish_public.sh            # publish
#   DRY_RUN=1 bash tools/publish_public.sh  # prepare and list, push nothing
#
# Only files Git tracks are copied, so ignored local files (the raw NSCISC
# data, the database) can never be published. The paths in EXCLUDE are left
# out on top of that. The public repository has its own history and has never
# contained the NSCISC data; the private one keeps the full history.
set -euo pipefail

PUBLIC_URL="${PUBLIC_URL:-https://github.com/hkm4lmn/SCI-RTW-public.git}"
EXCLUDE=(
  "processed"                                            # patient-level rows derived from NSCISC
  "PROPOSAL CSP600 HAKIM AL-AMIN 2025197013.pdf"         # personal document
  # Report files: their figures still show screenshots taken with an NSCISC
  # test patient. Remove these lines once the figures are refreshed.
  "docs/FYP Report - Combined.docx"
  "docs/FYP Report - Combined.pdf"
  "docs/Chapter 4 - Results and Discussion.docx"
  "docs/Chapter 5 - Conclusion and Recommendations.docx"
  "docs/UI Design - SCI-RTW Predict.docx"
)

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [ -n "$(git status --porcelain)" ]; then
  echo "Commit or stash your changes first: only committed work is published." >&2
  exit 1
fi
SHA="$(git rev-parse --short HEAD)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone -q "$PUBLIC_URL" "$WORK/public"
(cd "$WORK/public" && git rm -rq --ignore-unmatch .)
git archive HEAD | tar -x -C "$WORK/public"
for path in "${EXCLUDE[@]}"; do rm -rf "$WORK/public/$path"; done

cd "$WORK/public"
# Safety net: never publish a data or database file, whatever else changes.
if find . -path ./.git -prune -o -type f \( -iname '*.csv' -o -iname '*.sas7bdat' -o -iname '*.db' -o -iname '*.xlsx' \) -print | grep -q .; then
  echo "Refusing to publish: a data file is present:" >&2
  find . -path ./.git -prune -o -type f \( -iname '*.csv' -o -iname '*.sas7bdat' -o -iname '*.db' -o -iname '*.xlsx' \) -print >&2
  exit 1
fi
git add -A
if git diff --cached --quiet; then
  echo "The public repository is already up to date with $SHA."
  exit 0
fi
git diff --cached --stat | tail -1
if [ "${DRY_RUN:-}" = "1" ]; then
  echo "Dry run: nothing pushed. Files that would be public:"
  git ls-files | awk -F/ '{print $1}' | sort | uniq -c
  exit 0
fi
git -c user.name="$(git -C "$ROOT" config user.name)" -c user.email="$(git -C "$ROOT" config user.email)" \
  commit -q -m "Update from the private repository ($SHA)"
git -c http.postBuffer=524288000 push -q origin HEAD:main
echo "Published $SHA to $PUBLIC_URL"
