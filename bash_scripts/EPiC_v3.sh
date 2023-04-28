#!/bin/bash
python3 EPiC_code/train_val_split.py --scenario 3
wait

for i in 0 1 2 3 
do
    python3 EPiC_code/signal_main.py --scenario 3 --fold $i 
    wait 

    python3 EPiC_code/train.py --lr 0.0001 --epochs 30 --scenario 3 --fold $i --emotion valence --pretraining
    wait
    python3 EPiC_code/train.py --lr 0.0001 --epochs 30 --scenario 3 --fold $i --emotion arousal --pretraining
    wait

    python3 EPiC_code/train.py --lr 0.0001 --epochs 10 --scenario 3 --fold $i --emotion valence
    wait  
    python3 EPiC_code/train.py --lr 0.0001 --epochs 10 --scenario 3 --fold $i --emotion arousal 
    wait  
    python3 EPiC_code/train.py --lr 0.0001 --epochs 10 --scenario 3 --fold $i --emotion valence --use-pretrain
    wait
    python3 EPiC_code/train.py --lr 0.0001 --epochs 10 --scenario 3 --fold $i --emotion arousal --use-pretrain
    wait
done



