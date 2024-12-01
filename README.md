# Stanford AI Graduate Program: CS229-Final Project - Multitask and Multilingual Classifier

## Motivation:
One of the key challenges in building a multilingual classifier is ensuring consistent accuracy and fairness across languages. Variations in both quantity and quality of training data for different languages often lead to performance disparities. High-resource languages (like English) are classified more accurately compared to low-resource languages, making fairness and effectiveness across all languages difficult to achieve.

A promising approach to address this limitation has been identified as transfer learning through multi-task methods. It is hypothesized that a multilingual or even a monolingual model trained across multiple tasks could have its ability to understand toxicity in various languages enhanced. Specifically, given the scarcity of labeled toxic data in non-English languages, the model is proposed to be trained with other available annotated datasets for related tasks to improve its capacity to recognize toxicity across languages. The core assumption is that gathering a substantial amount of toxic and non-toxic annotated content in English is easier than in other languages, where doing so is highly time-consuming. Multilingual datasets designed for other tasks are suggested to be leveraged as an effective strategy to boost performance in multilingual toxicity detection, provided that a connection between the tasks can be identified.

## Research Question:
How can multi-task learning be used to combine available datasets and enhance performance on the target task of multilingual classification?

## How to run:
Train Multilingual classifier:
training script:model_training.py 
config: tran_config.yaml

Train Multilingual Multitask classifier:
training script: multitask_model_training.py 
config: tran_config.yaml

Evaluate classifiers:
script: model_extrinsic_evaluation.py
script: model_intrinsic_evalution.py - embedding evaluation.
config: eval_config.yaml