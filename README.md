# Pytorch-DCTTS
A Pytorch implementation of DCTTS ([Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969))

Note this is still WIP and I will update once the training is complete and working well.

## 1. Before Training
One should have a training and eval directory containing melspectrograms and linear spectrograms using tools such as
[Librosa](https://librosa.github.io/librosa/) to compute.

## 2. Train Text2Mel
To train text2mel module, run the following command `python text2mel.py`. Evaluation audio will be saved in local directory called `save_stuff/text2mel`. One can adjust the *batch_size* hyperparameter in the `train_text2mel.py` to maximize GPU usage.

## 3. Train SSRN
To train text2mel module, run the following command `python text2mel.py`. Evaluation audio will be saved in local directory called `save_stuff/ssrn`. One can adjust the *batch_size* hyperparameter in the `train_ssrn.py` to maximize GPU usage.


# Todo
1. batching
2. add regularization such as dropout, layer normalization, etc
3. improve UI/pipeline for loading data and generating custom setences
4. replace [lws](https://github.com/Jonathan-LeRoux/lws) vocoder with sampleRNN/fft-net/wavenet
