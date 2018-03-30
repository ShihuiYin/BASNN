# BASNN
Binary-Activation Spiking Neural Networks

Refer to https://arxiv.org/abs/1709.06206 for more details.

This paper presents a new back propagation based training algorithm for discrete-time spiking neural networks (SNN). Inspired by recent deep learning algorithms on binarized neural networks, binary activation with a straight-through gradient estimator is used to model the leaky integrate-fire spiking neuron, overcoming the difficulty in training SNNs using back propagation. Two SNN training algorithms are proposed: (1) SNN with discontinuous integration, which is suitable for rate-coded input spikes, and (2) SNN with continuous integration, which is more general and can handle input spikes with temporal information. Neuromorphic hardware designed in 40nm CMOS exploits the spike sparsity and demonstrates high classification accuracy (>98% on MNIST) and low energy (48.4-773 nJ/image).

To train an fully-connected or convolutional discrete-time SNN, just run "python train_SNNDC_MNIST_mlp.py" or "python train_SNNDC_MNIST_conv.py". More options could be found with "-h" appended. To test the trained SNN models, you can run "python test_SNNDC_MNIST_mlp.py" or "python test_SNNDC_MNIST_conv.py". More options could be found with "-h" appended. With increasing number of time steps per sample, the accuracy will improve. Compared to conventional SNN converted from ANN, the accuracy is quite high even with one or two time steps. More details are descirbed in the paper aforementioned.
