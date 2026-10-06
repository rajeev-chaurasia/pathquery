#!/usr/bin/env bash
# Fetch HotpotQA from HuggingFace.
#
# Not from curtis.ml.cmu.edu, which is where every tutorial points and which no
# longer answers at all. The HuggingFace copy is the same data in parquet.
#
# The distractor configuration is the one used here: it ships ten paragraphs
# per question, two of them containing the supporting sentences, and pooling
# those paragraphs across the dataset is what the corpus is built from.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
cd data/raw

BASE=https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor
for f in validation-00000-of-00001.parquet train-00000-of-00002.parquet train-00001-of-00002.parquet; do
  [ -f "$f" ] && { echo "have $f"; continue; }
  echo "fetching $f"
  curl -fL --retry 5 --retry-delay 2 -C - -o "$f" "$BASE/$f"
done
