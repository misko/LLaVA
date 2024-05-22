#!/bin/bash
set -eu

# Note: other servers are needed too; see:
# https://github.com/haotian-liu/LLaVA/blob/main/docs/LoRA.md

python \
  -m llava.serve.model_worker \
  --host 0.0.0.0 \
  --controller http://localhost:10000 \
  --port 40000 \
  --worker http://localhost:40000 \
  --model-path ~/meals-ds-1000-json/output/ \
  --model-name llava-v1.5-7b-test1234 \
  --model-base ../llava-v1.5-7b
