#!/bin/bash
python3 EPiC_code/train_test_split.py --scenario 1
wait 
python3 EPiC_code/signal_main.py --scenario 1 --final-flag
wait 

python3 EPiC_code/pretraining_final.py --lr 0.0001 --epochs 30 --scenario 1 --emotion valence --pretraining
wait
python3 EPiC_code/pretraining_final.py --lr 0.0001 --epochs 30 --scenario 1 --emotion arousal --pretraining
wait

python3 EPiC_code/test_final.py --lr 0.0001 --epochs 10 --scenario 1 --emotion valence
wait  
python3 EPiC_code/test_final.py --lr 0.0001 --epochs 10 --scenario 1 --emotion valence --use-pretrain
wait
python3 EPiC_code/test_final.py --lr 0.0001 --epochs 10 --scenario 1 --emotion arousal 
wait  
python3 EPiC_code/test_final.py --lr 0.0001 --epochs 10 --scenario 1 --emotion arousal --use-pretrain

