# Copyright 2017    Shihui Yin    Arizona State University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Description: Simulate Convolutional Spiking Neural Network (IF, reset by subtraction, MLP) using GPU by theano
from __future__ import division
import scipy.io as sio
from SNN_BA import ConvSNN_BA, SNN_BA_Config
import numpy as np
import sys
from pylearn2.datasets.mnist import MNIST
import argparse
from ast import literal_eval as bool

parser = argparse.ArgumentParser(description="BASNN MLP for MNIST", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-de', dest='deterministic', type=bool, default=False, help="If true, input is deterministic")
parser.add_argument('-sp', dest='save_path', default="MNIST_param_cnn.npz", help="Path to save parameters")
parser.add_argument('-bn', dest='batch_norm', type=bool, default=False, help="Batch normalization parameters applied")
parser.add_argument('-ns', dest='numStepsPerSample', type=int, default=1, help="Number of time steps each example is presented")
parser.add_argument('-qb', dest='quant_bits', type=int, default=0, help="Quantization bits of weights if greater than 0")
parser.add_argument('-ll', dest='last_layer', type=int, default=-1, help="Last layer to simulate (-1 means all layers will be simulated)")
parser.add_argument('-oz', dest='output_zero_threshold', type=bool, default=True, help="Threshold at output is set to zero instead of 1 if True")
parser.add_argument('-nr', dest='numRuns', type=int, default=20, help="Number of runs")
args = parser.parse_args()
print(args)

save_path = args.save_path
numStepsPerSample = args.numStepsPerSample
batch_norm = args.batch_norm
quant_bits = args.quant_bits
last_layer = args.last_layer
output_zero_threshold = args.output_zero_threshold
deterministic = args.deterministic
numRuns = args.numRuns

# load model from npz file

with np.load(save_path) as f:
    W_values = [f['arr_%d' % i] for i in range(len(f.files))]

n_layers = int(len(W_values) / 2)
nnsize = []
for i in range(n_layers):
    nnsize.append(W_values[2*i+1].shape[0])
print(nnsize)
# load test dataset
test_set = MNIST(which_set= 'test', center = False)
data = test_set.X
label = np.squeeze(test_set.y)
data = data.reshape((-1, 1, 28, 28))
W_SNN = W_values
unit_bias_steps = [1.] * n_layers

if quant_bits > 0: # quantization bits are provided
    print("Quantization to %d bits" % quant_bits)
    max_abs_W = 0
    for i in range(n_layers):
        temp_abs_W = np.amax(abs(W_SNN[i*2]))
        if temp_abs_W > max_abs_W:
            max_abs_W = temp_abs_W
    int_bits = np.ceil(np.log2(max_abs_W))+1
    print("Integer bits: %d bits" % int_bits)
    frac_bits = quant_bits - int_bits
    base = 2. ** frac_bits
    for i in range(len(W_SNN)):
        W_SNN[i] = np.round(W_SNN[i]*base) / base

# Set up SNN configuration
snn_config = SNN_BA_Config(snnsize=nnsize, threshold=1., initVmem=0., numStepsPerSample=numStepsPerSample)
snn_config.W = W_SNN
snn_config.batchsize = 10000
snn_config.unit_bias_steps = unit_bias_steps
snn_config.deterministic = deterministic
snn_config.output_zero_threshold = output_zero_threshold
snn_config.input_shape = data.shape

# set up ConvSNN
snn = ConvSNN_BA(snn_config)
snn.lastlayer = last_layer
if snn.lastlayer == -1:
    snn.lastlayer = n_layers

# Simulate ConvSNN
testErr = 0
if snn_config.deterministic == False:
    for i in range(numRuns):
        snn.sim(data, label)
        testErr += snn.testErr
    testErr = testErr/numRuns
    print("Average test error: %.2f%%" % (testErr*100))
else:
    snn.sim(data, label)
# Save the results
# print("Saving results...")
# dict = {'outputRaster': snn.outputRaster}
# sio.savemat('./data/results/tsnn_baseline_normed_%d.mat' % snn.lastlayer, dict)
