# AdversarialTraffic

This repository contains the implementation of adversarial examples of network traffic in website fingerprinting. We implemented four differenct adversarial example generation methods, like FGSM, DeepFool, PGD and advGAN, orginally introduced in image domain, we customized them to apply on network traffic.

** The data set and code for research purpose only**


## Content

This repository contains separate directories for the attacks, train, tools and nni_search. A brief description of the contents of these directories is below.  More detailed usage instructions are found in the individual directories' README.md.

The ```attack``` directory contains the code for generating adversarial examples with different algorithms.

The ```train``` directory contains the code for the deep learning based website fingerprinting models, GAN model, testing functions, data preparation, and some related utility functions.

The ```tools``` directory contains the code for data preprocessing, evaluations,  detecting adversary with different models.

The ```nni_search``` directory contains the code for hyper-perameters search for CNN and LSTM models by using NNI tool.


### Datasets

The dataset we used is collected by Sirinam et.al. It includes 95 monitored websites with 1,000 traces per website for the closed world evaluation and 40,716 unmonitored websites with 1 trace per website for the open-workd evaluation. You can find more details here "https://github.com/deep-fingerprinting/df" .


## Requirements

This project is entirely written in Python 3, based on PyTorch 1.3.0. We recommend you to run it with a GPU machine.


## Usage

See the project's directories for usage information.

## Citation
When reporting results that use the dataset or code in this repository, please cite:

Hao Liu, Jimmy Dani, Hongkai Yu, Wenhai Sun, Boyang Wang, "AdvTraffic: Obfuscating Encrypted Traffic with
Adversarial Examples", IEEE/ACM International Symposium on Quality of Service (**IEEE/ACM IWQoS 2022**), June, 2022.

## Contacts
Hao Liu, liu3ho@mail.uc.edu

Boyang Wang, boyang.wang@uc.edu
