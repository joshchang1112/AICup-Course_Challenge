# AICup Course Challenge - Thesis Classification

## Competition description

The aim of this competition is to attempt to solve a problem that has troubled researchers for long: 
“How do you design a system that could automatically analyze the abstract of a thesis and summarize utilized, compared, or 
newly proposed algorithm used within the theses?”

The contestants will be provided theses with the topic of Computer Science sourced from arXiv. 
The contestants should use the provided materials to predict if a thesis should be classified as the following categories: 
Theoretical, Engineering, Empirical, or Others. Note that a thesis may have multiple classifications, 
e.g. a sentence may be classified as both Theoretical and Engineering.

Competition website: https://tbrain.trendmicro.com.tw/Competitions/Details/12

## Score Leaderboard
Team Name: 公鹿總冠軍 (Milwaukee Bucks Champion:trophy:)

Public Score:

0.729490 (Rank:10/148)

Private Score:

0.725064 (Rank:**4**/148)

## Installation

To execute our code successfully, you need to install Python3 and PyTorch (our deep learning framework) first. 
Please refer to [Python installing page](https://www.python.org/downloads/) and [Pytorch installing page](https://pytorch.org/get-started/locally/#start-locally) 
regarding the specific install command for your platform.

When PyTorch has been installed, other packages can be installed using pip as follows:
```
pip3 install -r requirement.txt
```

## Code organization

*   `preprocess.py`: Tokenize the raw texts of papers, convert word tokens to ids for below training using RoBERTa.

*   `train.py`: Fine-tuned RoBERTa model on multi-label classification task with args parameters. `python train.py --help` for more information. 

*   `dataset.py`: The dataset when fine-tuning RoBERTa model.

*   `metrics.py`: Calculate F1 score for training.

*   `utils.py`: Data processing utils functions.

*   `ensemble.py`: Ensemble all results by average voting.

## Code Usage

1. Preprocess, fine-tuned, predict by using RoBERTa pretrained model.
```
python3 train.py
```
2. Ensemble the results.
```
python3 ensemble.py
```

## Contact information

For help or issues using our code, please contact Sung-Ping Chang (`joshspchang@gmail.com`).
