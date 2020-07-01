# Cross-Lingual Transfer Learning

## Language Adverserial Network for Cross-Lingual Text Classification

#### Source code: [Language-Adversarial Training for Cross-Lingual Text Classification (TACL)](https://github.com/ccsasuke/adan "Source code on Github")

## Introduction

__Language-Adversarial Training__ technique for the cross-lingual model transfer problem learns a language-invariant hidden feature space to achieve better cross-lingual generalization.

In this project, we present our language-adversarial training approach in the context of __Cross-lingual text classification (CLTC)__

We classify a product review into five categories corresponding to its star rating. 

<a href="https://camo.githubusercontent.com/4d97fa1a94cef8a74c708fba238c65a4b30db47e/687474703a2f2f7777772e63732e636f726e656c6c2e6564752f7e786c6368656e2f6173736574732f696d616765732f6164616e2e706e67"><img src="https://camo.githubusercontent.com/4d97fa1a94cef8a74c708fba238c65a4b30db47e/687474703a2f2f7777772e63732e636f726e656c6c2e6564752f7e786c6368656e2f6173736574732f696d616765732f6164616e2e706e67" width="250" height="400"> </a>

LAN has two branches. There are four main components in the network: 
- Embedding averager __EA__ that maps input sequence x to embeddings and gives an averaging of the embeddigs of the sequence.
- Joint Feature extractor __F__ that maps the averaged embeddings to a fixed-length feature vector in the shared feature space.
- Sentiment classifier __P__ that predicts the label for x given the feature representation F (x).
- Language discriminator __Q__ that also takes F (x) but predicts a scalar score indicating whether x is from SOURCE (1) or TARGET (-1).

We adopt the __Deep Averaging Network (DAN)__ for the (EA + F). DAN takes the arithmetic mean of the word vectors as input, and passes it through several fully-connected layers until a softmax for classification.

EA takes the arithmetic mean of the word vectors as input, F passes it through several fully-connected layers until a softmax for classification in Semantic Classifier (P). In LAN, EA first calculates the average of the word vectors in the input sequence, then passes the average through a feed-forward network with ReLU nonlinearities. The activations of the last layer in F are considered the extracted features for the input and are then passed on to P and Q. The sentiment classifier P and the language discriminator Q are standard feed-forward networks. P has a softmax layer on top for text classification and Q ends with a tanh layer of output width 1 to assign a language identification score (-1 for target and 1 for source).

To run the program on local system: 

The directory should be: 'My Drive/Colab Notebooks/Work/'

In that two directories get created after running extract_data.ipynb:

-> 'My Drive/Colab Notebooks/Work/Amazon reviews/'
>> 'My Drive/Colab Notebooks/Work/Amazon reviews/train'

>> 'My Drive/Colab Notebooks/Work/Amazon reviews/dev'

>> 'My Drive/Colab Notebooks/Work/Amazon reviews/test'

-> 'My Drive/Colab Notebooks/Work/bwe/'
>> 'My Drive/Colab Notebooks/Work/bwe/vectors'
