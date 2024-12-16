#!/bin/bash


COMMAND="poetry run python eval_perplexity_wt.py --model_dir llama_3.2-3B_model/original/ --num_samples 2000 --sae_dir sae_final_2024-10-28.pth --drop"

for i in {0..100..10}; do
	    echo "Executing: $COMMAND $i"
	        $COMMAND $i
	done
