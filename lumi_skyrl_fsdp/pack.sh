#!/usr/bin/env bash
# Usage: bash pack.sh <dir> > unpack.sh
set -euo pipefail

cd "${1:-.}"
echo '#!/usr/bin/env bash'
echo 'set -euo pipefail'

find . -type f ! -path './.git/*' ! -path '*/__pycache__/*' -print0 | sort -z |
while IFS= read -r -d '' f; do
  # Skip binary files (empty files pass)
  if [ -s "$f" ] && ! grep -Iq . "$f"; then
    echo "skip binary: $f" >&2
    continue
  fi
  f="${f#./}"
  # Unique heredoc delimiter per file to avoid collisions with content
  d="EOF_$(md5sum <<<"$f" | cut -c1-12)"
  printf 'mkdir -p %q\n' "$(dirname "$f")"
  printf "cat > %q <<'%s'\n" "$f" "$d"
  cat "$f"
  # Ensure delimiter starts on its own line
  if [ -n "$(tail -c1 "$f")" ]; then echo; fi
  echo "$d"
  if [ -x "$f" ]; then printf 'chmod +x %q\n' "$f"; fi
done