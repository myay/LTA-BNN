#! /bin/bash

# error_probs="1E-2 2E-2"
error_probs="1E-2 2E-2 5E-2 1E-1"

for error_prob in ${error_probs}
do
  # for j in {64..72..4}
  for j in {4..256..4}
  do
    # echo ${error_prob}
    # echo ${j}
    python3 run.py --model=VGG3 --dataset=FMNIST --load-model-path=models/traditional_training/fmnist_cnn_mhl_100ep.pt --tlu-mode=1 --nr-xnor-gates=${j} --error-prob=${error_prob}
  done
done
