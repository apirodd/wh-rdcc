#!/usr/bin/env bash
# scripts/download_dichasus.sh
# Download completo del dataset DICHASUS cf0x da DaRUS
# Uso: bash scripts/download_dichasus.sh

set -e

BASE_API="https://darus.uni-stuttgart.de/api/access/datafile"
BASE_WEB="https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x"

mkdir -p data_raw
cd data_raw

echo "=== Download DICHASUS cf0x ==="
echo "Dimensioni attese: cf02=4.9GB, cf03=5.9GB, cf04=11.1GB"
echo ""

# TFRecords via DaRUS API
for FILE_ID in "14:dichasus-cf02.tfrecords" \
               "15:dichasus-cf03.tfrecords" \
               "16:dichasus-cf04.tfrecords"; do
    ID="${FILE_ID%%:*}"
    FNAME="${FILE_ID##*:}"
    if [ -f "$FNAME" ] && [ "$(stat -c%s "$FNAME" 2>/dev/null || stat -f%z "$FNAME")" -gt 500000000 ]; then
        echo "  OK (già presente): $FNAME"
    else
        echo "  Download: $FNAME ..."
        wget -q --show-progress \
             -O "$FNAME" \
             "${BASE_API}/:persistentId?persistentId=doi:10.18419/DARUS-2854/${ID}"
    fi
done

# JSON files via sito DICHASUS (no SSL verify per certificato scaduto)
for FNAME in "reftx-offsets-dichasus-cf02.json" \
             "reftx-offsets-dichasus-cf03.json" \
             "reftx-offsets-dichasus-cf04.json"; do
    if [ -f "$FNAME" ] && [ "$(stat -c%s "$FNAME" 2>/dev/null || stat -f%z "$FNAME")" -gt 100 ]; then
        echo "  OK (già presente): $FNAME"
    else
        echo "  Download: $FNAME ..."
        wget -q --no-check-certificate \
             -O "$FNAME" \
             "${BASE_WEB}/${FNAME}"
    fi
done

echo ""
echo "=== Contenuto data_raw/ ==="
ls -lh .
echo ""
echo "Ora esegui: python src/preprocessing.py"
