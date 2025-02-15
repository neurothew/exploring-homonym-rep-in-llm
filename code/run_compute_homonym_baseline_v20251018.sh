#!/bin/bash

# Models by family
# models=("Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B")
# models=("FacebookAI/roberta-base" "FacebookAI/roberta-large" "xlnet/xlnet-base-cased" "xlnet/xlnet-large-cased" "microsoft/deberta-v3-base" "microsoft/deberta-v3-large")
# models=("bert-base-uncased" "bert-large-uncased")
# models=("meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B")
# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")

# English models
# models=("bert-base-uncased" "bert-large-uncased" "FacebookAI/roberta-base" "FacebookAI/roberta-large" "FacebookAI/xlm-roberta-base" "FacebookAI/xlm-roberta-large" "microsoft/deberta-v3-base" "microsoft/mdeberta-v3-base" "microsoft/deberta-v3-large" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B")

# Chinese models
# models=("bert-base-chinese" "FacebookAI/roberta-base" "FacebookAI/roberta-large" "FacebookAI/xlm-roberta-base" "FacebookAI/xlm-roberta-large" "microsoft/deberta-v3-base" "microsoft/mdeberta-v3-base" "microsoft/deberta-v3-large" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B")

# Multilingual
models=("bert-base-multilingual-uncased")

for model in "${models[@]}"
do
    # ----- Compute Chinese homonym baseline -----
    # ----- Different sense -----
    # echo "Running: python compute_baseline_sense-diff_lang-agnostic.py -m $model -l zh"
    # python compute_baseline_sense-diff_lang-agnostic.py -m "$model" -l "zh"
    # echo "----------------------------------------"

    # ----- Same sense -----
    # echo "Running: python compute_baseline_sense-same_lang-agnostic.py -m $model -l zh"
    # python compute_baseline_sense-same_lang-agnostic.py -m "$model" -l "zh"
    # echo "----------------------------------------"

    # ----- Compute English homonym baseline -----
    # ----- Different sense -----
    # echo "Running: python compute_baseline_sense-diff_lang-agnostic.py -m $model -l en"
    # python compute_baseline_sense-diff_lang-agnostic.py -m "$model" -l "en"
    # echo "----------------------------------------"

    # ----- Same sense -----
    echo "Running: python compute_baseline_sense-same_lang-agnostic.py -m $model -l en"
    python compute_baseline_sense-same_lang-agnostic.py -m "$model" -l "en"
    echo "----------------------------------------"

    # echo "Running: python compute_baseline_sense-same.py -m $model"
    # python compute_baseline_sense-same.py -m "$model"
    # echo "----------------------------------------"
done