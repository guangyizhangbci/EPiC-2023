mkdir -p /path/to/folder/results/scenario_1/test/annotations/
wait
scp -r /path/to/folder/EPiC_result/final/ecg/retrain/scenario_1/*   /path/to/folder/results/scenario_1/test/annotations/
wait

for i in 0 1 2 3 4
do 
    mkdir -p /path/to/folder/results/scenario_2/fold_$i/test/annotations/
    wait
    scp -r /path/to/folder/EPiC_result/final/ecg/retrain/scenario_2/fold_$i/*   /path/to/folder/results/scenario_2/fold_$i/test/annotations/
    wait
done

for i in 0 1 2 3 
do 
    mkdir -p /path/to/folder/results/scenario_3/fold_$i/test/annotations/
    wait
    scp -r /path/to/folder/EPiC_result/final/ecg/retrain/scenario_3/fold_$i/*   /path/to/folder/results/scenario_3/fold_$i/test/annotations/
    wait
done

for i in 0 1 
do 
    mkdir -p /path/to/folder/results/scenario_4/fold_$i/test/annotations/
    wait    
    scp -r /path/to/folder/EPiC_result/final/ecg/retrain/scenario_4/fold_$i/*  /path/to/folder/results/scenario_4/fold_$i/test/annotations/
    wait
done


