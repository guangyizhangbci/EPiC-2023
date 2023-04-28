# EPiC-2023: A Simple End-to-end Deep Learning solution for Emotion Recognition using Physiological Signals


Team: Guangyi Zhang and Ali Etemad 



- [x] Complete pipeline
- [x] Preprocessing of ECG and GSR
- [ ] Preprocessing of more modalities
- [ ] Multimodal 

# Files 

parent_dir/
│
├── EPiC_code/ (source code directory)
│   ├── ecg_net.py (end-to-end deep learning modules: Conv1D and Transformer)
│   ├── parsing.py 
│   ├── preprocessing.py 
│   ├── pretraining_final.py 
│   ├── signal_main.py
│   ├── test_final.py
│   ├── train.py
│   ├── train_test_slit.py
│   ├── train_val_slit.py
│   └── utils.py
│
├── bash scripts/ (comands to generate val/test results for each scenario)
│   ├── Epic_s1.py (Scenario 1: experiment on test set)
│   ├── Epic_s2.py (Scenario 2 (5 folds): experiment on test set)
│   ├── Epic_s3.py (Scenario 3 (4 folds): experiment on test set)
│   ├── Epic_s4.py (Scenario 4 (2 folds): experiment on test set)
│   ├── Epic_v1.py (Scenario 1: experiment on validation set)
│   ├── Epic_v2.py (Scenario 2 (5 folds): experiment on validation set)
│   ├── Epic_v3.py (Scenario 3 (4 folds): experiment on validation set)
│   └── Epic_v4.py (Scenario 4 (3 folds): experiment on validation set)
│
├── README.md (documentation file with project information)










# Statement of Limitations

Please note that this project was developed under specific constraints which may have affected the performance and generalization of the model. Our team was limited to only two members, with me spending less than 40 hours to form this version of code or solution. Additionally, I had very limited GPU resources during the development process.

Despite these constraints, I have built a complete pipeline that can be utilized by other researchers to build upon and potentially improve the model's performance. We encourage the community to explore and experiment with this pipeline to optimize the results further.

In this project, we focused on single channel ECG out of a total of 8 available signales including other modalities. However, we believe that incorporating additional modalities could lead to better results. Our code is designed to be easily extendable, and we welcome researchers to modify and adapt it to their specific needs.

We appreciate any contributions, suggestions, or improvements that others might offer to enhance the performance and applicability of our work. Please feel free to fork this repository, submit pull requests, or open issues to discuss potential improvements. 
