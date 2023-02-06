#! /bin/bash

error_probs="1E-2 2E-2 5E-2 1E-1"

for error_prob in ${error_probs}
do
  python3 run.py --model=VGG3 --dataset=FMNIST --train-model=1 --epochs=100 --lr=0.001 --step-size=10 --batch-size=256 --tlu-train=1 --tlu-mode=1 --nr-xnor-gates=64 --error-prob=${error_prob} --save-model="FMNIST_TLU_TRAIN_NOISE_${error_prob}"
done
