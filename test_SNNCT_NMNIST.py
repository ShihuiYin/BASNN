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

# Description: Simulate Spiking Neural Network (IF, reset by subtraction, MLP) using GPU by theano
# Created on 03/04/2017
# Modified on 03/11/2017, normalize input
# Modified on 03/12/2017, for MNIST
# Modified on 04/03/2017, with BASNN weights
# Modified on 04/20/2017, with BASNN temporal coding weights
# Modified on 04/27/2017, with BASNN temporal coding weights for NMNIST
# Modified on 05/03/2017, with motion detection added
# Modified on 06/23/2017, one-hot coded motion detection neurons

import scipy.io as sio
from tSNN import tSNN, tSNN_Config
import numpy as np
import sys
import gzip, pickle
import h5py
import sys

# load model from npz file
numStepsPerSample = int(sys.argv[1])
param_path = sys.argv[2]
with np.load(param_path) as f:
    W_values = [f['arr_%d' % i] for i in range(len(f.files))]

n_layers = int(len(W_values) / 2)
nnsize = []
W = []
for i in range(n_layers):
    nnsize.append(W_values[2*i].shape[0])
    W.append(np.vstack((W_values[2*i+1].reshape(1,-1), W_values[2*i])))
nnsize.append(W_values[-1].shape[0])
print(nnsize)
# import pdb; pdb.set_trace()
# load test dataset
test_set = h5py.File('/home/syin11/pythonsim/BASNN/data/NMNIST-Test_%d.mat' % numStepsPerSample)
test_set_x = test_set['Data']
test_set_x = test_set_x[()]
test_set_x = np.swapaxes(test_set_x, 0, 2).astype('float32')
test_set_y = test_set['labels']
test_set_y = test_set_y[()].transpose().astype('int8')
test_set_y = np.hstack(test_set_y)
test_set_y1 = np.stack((test_set_y, np.zeros((10000,),dtype='int8')), axis=1)
test_set_y2 = np.stack((test_set_y, np.ones((10000,),dtype='int8')), axis=1)
test_set_y = np.concatenate((test_set_y1, test_set_y2), axis=0)
test_set_x1 = test_set_x
test_set_x2 = np.flipud(test_set_x)
test_set_x = np.concatenate((test_set_x1, test_set_x2), axis=1)
# import pdb; pdb.set_trace()
W_SNN = W

if len(sys.argv) > 3 and int(sys.argv[3]) != 0: # quantization bits are provided
    quant_bits = int(sys.argv[3])
    print("Quantization to %d bits" % quant_bits)
    max_abs_W = 0
    for i in range(n_layers):
        temp_abs_W = np.amax(abs(W_values[i*2]))
        if temp_abs_W > max_abs_W:
            max_abs_W = temp_abs_W
    int_bits = np.ceil(np.log2(max_abs_W))+1
    print("Integer bits: %d bits" % int_bits)
    frac_bits = quant_bits - int_bits
    base = 2. ** frac_bits
    for i in range(len(W_SNN)):
        W_SNN[i] = np.round(W_SNN[i]*base) / base

unit_bias_steps = [1., 1., 1.]
snn_config = tSNN_Config(snnsize=nnsize, threshold=1.0, initVmem=0., numStepsPerSample=16)
snn_config.W = W_SNN
snn_config.batchsize = 20000
snn_config.unit_bias_steps = unit_bias_steps
snn_config.input_coding = 'identity'
# snn_config.output_decoding = 'first-to-spike'
snn_config.output_decoding = 'spike-count'
# snn_config.output_decoding = 'last-step'
snn_config.motion_detect = True
snn_config.motion_index = -2
snn_config.motion_decoding = 'one-hot'
snn_config.output_zero_threshold = False
snn_config.output_memoryless = False

# set up SNN
snn = tSNN(snn_config)
snn.lastlayer=3

# Simulate SNN
snn.sim(test_set_x, test_set_y)

# Save the results
print("Saving results...")
dict = {'outputRaster': snn.outputRaster}
sio.savemat('./data/results/tsnn_nmnist_%d.mat' % snn.lastlayer, dict)
# import pdb; pdb.set_trace()
