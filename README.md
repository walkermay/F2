# Code for F2

## Prerequisites

Name and versions of Libraries

**Python** 3.10.18,

**Pytorch** 2.3.0,

**torchvision** 0.18.0,

**Numpy** 1.26.4,

**tqdm** 4.67.1.


## Datasets

We use the following three datasets in our experiments.

HotpotQA dataset: https://huggingface.co/datasets/hotpotqa

MS-MARCO dataset: https://huggingface.co/datasets/ms_marco

NQ dataset: https://huggingface.co/datasets/natural_questions

Download the dataset from public resources and put them into datasets folder.

You could also run these lines to manually download datasets.

```
cd preprocess
python prepare-dataset.py
```

## Code structure

+ 1-1-baseline-answer.py: for obtaining the correct answer to the problem.
+ 1-2-unlearning-coarse.py: for achieving the CGU part in F2.
+ 1-2-unlearning-fine.py: for achieving the FGU part in F2.
+ 2-1-attribution.py: for finding influential documents.
+ 2-2-sort.py: for classifying influential documents.
+ 2-3-attack-coarse.py: for implementing CGU based attack in F2.
+ 2-3-attack-fine.py: for implementing FGU based attack in F2.
+ 2-3-attack-random-coarse.py: for implementing Rand+CGU based attack in F2. 
+ 2-3-attack-random-fine.py: for implementing Rand+FGU based attack in F2. 
+ datasets folder: for placing different datasets.
+ readme.md: for understanding the components of the code.

## Implement F2

Each script is designed to be run independently to perform its specific function. To execute a step, simply run the corresponding python file from your terminal.

For example, to run the attribution step to find influential documents, you would execute:

```
python 2-1-attribution.py
```

Similarly, to run any other function, such as `1-1-baseline-answer.py` or `2-3-attack-fine.py`, execute them in the same manner.

**Important:** All parameters are set directly within each script file. Before running a file, you may need to open it and modify variables such as `topk`, input/output paths, or the llm's JSON configuration path to fit your setup.

