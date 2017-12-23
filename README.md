# Signature Embedding
A Deep Metric Learning Approach to Signature Verification

## About this Project
The motivation of this work and the approach taken can be found in [Signature Embedding: Writer Independent Offline Signature Verification with Deep Metric Learning](https://hpi.de/fileadmin/user_upload/fachgebiete/meinel/tele-task/papers/isvc16_Hannes.pdf).

This project is an effort to re-implement the system presented in the paper, leveraging the knowledge I gained in the previous project and the technological advances since then, for example in the Chainer deep learning framework.

## Current State of the Project
Most of the basic functionality required to train the embedding network is implemented. To get startet, best take a look at `train_mnist.py`. The script is able to embed images from the MNIST dataset in the same way it should be done with handwritten signatures.
If you want to train your own signature embedding, you'll need to implement an appropriate network architecture (the MLP in the script won't do for signatures) and load your training data. Feel free to open an issue if you need any help.
