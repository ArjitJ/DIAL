# DIAL

Implementation of

- Deep Indexed Active Learning for Matching Heterogeneous Entity Representations

![](Images/TraditionalAL.png)
Traditional methods for Active Learning Pairwise classification tasks follow a pipeline as described: 
In each iteration, the learning algorithm (learner) learns a matcher (shown in an ellipse which we use to denote model components) from labeled data 𝑇,
the labeled pairs collected from the (human) labeler so far, while the example selector (selector) chooses the most informative unlabeled
pairs to acquire labels for. After including the new labels into 𝑇, the process repeats until we learn a matcher of sufficient quality.

![](Images/DIAL.png)
Our proposed integrated matcher-blocker combination and new AL workflow as shown. Compared to the previous diagram, the two most notable differences are 
1) the blocker (dashed box) is now part of the AL feedback loop, and 
2) the matcher is a component within the blocker. 
As base matcher, we use transformer-based pretrained language models (TPLM) which have recently led to excellent ER accuracies in the passive (non-AL) settings.

## Getting Started

### Environment
This code has been tested on a machine with 64 2.10GHz Intel Xeon Silver 4216 CPUs with 1007GB RAM and a single NVIDIA Titan Xp 12 GB GPU with CUDA 10.2 running Ubuntu 18.04

### Reproducing the Experiments

The first step is to get the data. We provide the data used in DeepMatcher experiments. The multilingual data can be downloaded from [salesforce/localization-xml-mt](https://github.com/salesforce/localization-xml-mt)

Now create a virtual environment using conda

```
conda env create -n DIAL_env -f environment.yml

```

Use run_expts.sh to replicate experiments from the paper. Example 

```
bash run_expts.sh DIAL amazon_google_exp 

```

Currently supports : Walmart-Amazon, Amazon-Google, DBLP-ACM, DLBP-Google Scholar, Abt-Buy

To run experiments with the multilingual dataset, 

```
cd MultiLingual
bash run_multilingual_expts.sh DIAL-Multilingual

```
