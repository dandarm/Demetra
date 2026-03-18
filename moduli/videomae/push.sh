#!/usr/bin/env bash
set -u

MSG="${1:-mylogs}"

echo "==> git add -A"
git add -A || exit 1

echo "==> git commit"
if git diff --cached --quiet; then
    echo "Nessuna modifica da committare."
else
    git commit -m "$MSG" || exit 1
fi

echo "==> git push con retry"
max_attempts=5
sleep_seconds=10

for ((i=1; i<=max_attempts; i++)); do
    echo "Tentativo push $i/$max_attempts..."
    if git push; then
        echo "Push completato con successo."
        exit 0
    fi

    echo "Push fallito."
    if [[ $i -lt $max_attempts ]]; then
        echo "Aspetto ${sleep_seconds}s prima di riprovare..."
        sleep "$sleep_seconds"
    fi
done

echo "Push fallito dopo $max_attempts tentativi."
echo "Diagnostica rapida:"
getent hosts github.com || true
curl -I https://github.com 2>/dev/null | head -n 5 || true

exit 1