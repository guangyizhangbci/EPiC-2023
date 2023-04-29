# EPiC-2023: A Naive End-to-End Deep Learning Approach for Emotion Recognition using Physiological Signals


Team: Guangyi Zhang and Ali Etemad 



- [x] Complete pipeline
- [x] End-to-end deep learning modules (convolutional neural network and transformer)
- [x] Preprocessing of ECG and GSR
- [ ] More end-to-end deep learning frameworks
- [ ] Multimodal learning



## Files 


```shell
parent_dir/
│
├── EPiC_code/                # source code directory)
│   ├── ecg_net.py            # end-to-end deep learning modules: Conv1D and Transformer
│   ├── parsing.py            # command line arguments (e.g., learning rate, choice of signal modalities, scenario numbers, fold numbers)
│   ├── preprocessing.py      # preprocessing for ECG and GSR
│   ├── pretraining_final.py  # pretraining using entire training data from the same fold under the same scenario
│   ├── signal_main.py        # filtering data and segmenting data with sliding windows
│   ├── test_final.py         # generation of test results with two options: 1. training from scratch and 2. retraining using the pre-trained model weights
│   ├── train.py              # obtaining validation results with two options: 1. training from scratch and 2. retraining using the pre-trained model weights
│   ├── train_test_slit.py    # train/test split
│   ├── train_val_slit.py     # train/val split with the split startegy according to train/test split one, random seed fixed
│   └── utils.py              # containing frequently used helper functions
│
├── bash scripts/             # comands to generate val/test results for each scenario
│   ├── EPiC_s1.sh            # scenario 1:           experiment on test set 
│   ├── EPiC_s2.sh            # scenario 2 (5 folds): experiment on test set
│   ├── EPiC_s3.sh            # scenario 3 (4 folds): experiment on test set
│   ├── EPiC_s4.sh            # scenario 4 (2 folds): experiment on test set
│   ├── EPiC_v1.sh            # scenario 1:           experiment on validation set
│   ├── EPiC_v2.sh            # scenario 2 (5 folds): experiment on validation set
│   ├── EPiC_v3.sh            # scenario 3 (4 folds): experiment on validation set
│   ├── EPiC_v4.sh            # scenario 4 (3 folds): experiment on validation set
│   └── submission.sh         # move path to the one required by the organizers
│
├── submissions/              # results
│   └── results.zip           # containing results in .csv format for all scenarios
│
├── README.md                 # documentation file with project information
│
├── requirements.txt          # required libs and their versions.

```


## Pipeline 
1. Data Preprocessing: This step entails cleansing and preparing the data for analysis, which includes filtering, normalization, and segmentation of physiological signals such as ECG and GSR. We opt for a window length of 10 seconds.

2. Model Selection: Next, we select end-to-end deep learning-based regression models (Convolutional Neural Network and Transformer) to predict valence and arousal ratings based on the acquired ECG representations.

3. Train/Validation Split: We adhere to the train-test split protocols provided by the organizer, splitting the validation set from the training set in each scenario.

4. Training strategies: 1) training from scratch, and 2) pretraining the model on the entire training data from the same fold in the same scenario, followed by retraining the model.

5. Evaluation: In scenario 1, we conduct subject-dependent and video session-dependent experiments. In each fold of scenario 2, we carry out session-dependent experiments. In each fold of both scenarios 3 and 4, we perform subject-dependent experiments.

6. Final results: We assess several baselines on the validation set and select the best one (according to averaged validation RMSE values) for the test set to obtain the final results for submission.

## To Do List. 
1. Fusion Method: After evaluating the performance of the model on each unimodal physiological signal, we will develop an efficient fusion method to integrate information from multiple physiological signals. This process might involve merging features extracted from various signals or combining the predictions of multiple models trained on different signals.

2. Multimodal Data Analysis: Ultimately, we will apply the entire pipeline to the multimodal data, where we will preprocess, extract features, select models, fine-tune hyperparameters, evaluate the models, and devise fusion methods for incorporating multiple signals. We will once again assess our model's performance using the RMSE metric and compare its performance with the unimodal models.

## Usage
The usage of this code is straightforward. The only modification needed is to change the ['/path/to/folder'](./EPiC_code/utils.py#LL12C2-L12C32) in 'utils.py' and any bash files you wish to use. You can opt to acquire validation results or generate test annotations with the bash files provided in the ['bash_scripts'](./bash_scripts) folder. Finally, if you want to generate test annotations in the format required by the organizers, please refer to the example in ['submission.sh'](./bash_scripts/submission.sh).


## Statement of Limitations

Please note that this project was developed under specific constraints which may have affected the performance and generalization of the model. Our team was limited to only two members, wih me dedicating weekends only with 40 hours in total to form this version of code or solution. Additionally, I had limited GPU resources during the development process.

Despite these constraints, I have built a complete pipeline that can be utilized by other researchers to build upon and potentially improve the model's performance. We encourage the community to explore and experiment with this pipeline to optimize the results further.

In this project, we focused on single channel ECG out of a total of 8 available signals including other modalities. However, we believe that incorporating additional modalities could lead to better results. Our code is designed to be easily extendable, and we welcome researchers to modify and adapt it to their specific needs.

We appreciate any contributions, suggestions, or improvements that others might offer to enhance the performance and applicability of our work. Please feel free to fork this repository, submit pull requests, or open issues to discuss potential improvements. 

Should you have any questions, please feel free to contact me at guangyi.zhang@queensu.ca.


[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

