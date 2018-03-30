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

# Description: Define SNN_BA class (with probabilistic input layer)
from __future__ import division
import numpy as np
import theano
import theano.tensor as T
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import collections


class mlp(object):
    """ mlp layer for spiking neural network """
    
    def __init__(self, input, n_in, n_out, W, unit_bias_step, threshold, initVmem=0.5, batchsize=1000):
        if input.ndim > 2:
            self.input = input.flatten(2)
        else:
            self.input = input # input spikes, shape = (batchsize, n_in), can be 0, 1, -1(for signed input layer outputs)
        self.n_in = n_in
        self.n_out = n_out
        self.W = W # weights and biases, shape = (n_in+1, n_out), the first row is bias
        self.unit_bias_step = unit_bias_step
        self.threshold = threshold
        self.initVmem = initVmem
        self.batchsize = batchsize
        self.Vmem_init = np.zeros((self.batchsize, self.n_out), dtype=theano.config.floatX) + np.float32(self.initVmem * threshold)
        self.Vmem = theano.shared(self.Vmem_init, name='Vmem', borrow=True)
        self.Vmem_update_prespike = self.Vmem + T.dot(self.input, self.W[1:]) + unit_bias_step * self.W[0]
        self.output = T.ge(self.Vmem_update_prespike, self.threshold)
        self.Vmem_update_postspike = T.cast(self.Vmem_update_prespike - self.output * self.threshold, theano.config.floatX)
        self.output_shape = (batchsize, n_out)
    def refresh(self):
        self.Vmem.set_value(self.Vmem_init)
        
class conv(object):
    """ conv layer for spiking neural network, by default, followed by a 2x2 max pooling layer"""
    def __init__(self, input, input_shape, W, b, unit_bias_step, threshold, initVmem=0.5, batchsize=1000, border_mode = 'valid', maxpooling = True):
        self.input = input
        self.input_shape = input_shape
        self.W = W
        self.b = b
        self.filtersize = W.shape[2:4]
        self.border_mode = border_mode
        self.maxpooling = maxpooling
        if self.border_mode == 'valid':
            mapsize_x_conv = input_shape[2]-self.filtersize[0]+1
            mapsize_y_conv = input_shape[3]-self.filtersize[1]+1
        elif self.border_mode == 'half':
            mapsize_x_conv = input_shape[2]
            mapsize_y_conv = input_shape[3]
        else:
            mapsize_x_conv = input_shape[2]+self.filtersize[0]-1
            mapsize_y_conv = input_shape[3]+self.filtersize[1]-1
        if self.maxpooling == True:
            mapsize_x = int(mapsize_x_conv/2)
            mapsize_y = int(mapsize_y_conv/2)
        else:
            mapsize_x = mapsize_x_conv
            mapsize_y = mapsize_y_conv
        self.output_shape_conv = (batchsize, W.shape[0], mapsize_x_conv, mapsize_y_conv)
        self.output_shape = (batchsize, W.shape[0], mapsize_x, mapsize_y) 
        self.unit_bias_step = unit_bias_step
        self.threshold = threshold
        self.initVmem = initVmem
        self.batchsize = batchsize
        print(self.input_shape)
        self.Vmem_init = np.zeros(self.output_shape_conv, dtype=theano.config.floatX) + np.float32(self.initVmem * threshold)
        self.Vmem = theano.shared(self.Vmem_init, name='Vmem', borrow=True)
        convolved = T.nnet.conv2d(input, T.as_tensor_variable(self.W), self.input_shape, self.W.shape, subsample=(1,1), border_mode=self.border_mode, filter_flip=True)
        self.Vmem_update_prespike = self.Vmem + convolved + unit_bias_step * T.as_tensor_variable(np.float32(self.b)).dimshuffle(('x',0,'x','x'))
        self.output_conv = T.cast(T.ge(self.Vmem_update_prespike, self.threshold), theano.config.floatX)
        self.Vmem_update_postspike = T.cast(self.Vmem_update_prespike - self.output_conv * self.threshold, theano.config.floatX)
        if self.maxpooling == True:
            self.output = T.signal.pool.pool_2d(self.output_conv, ds=(2, 2), st=None, ignore_border=True, padding=(0, 0), mode='max')
        else:
            self.output = self.output_conv
    def refresh(self):
        self.Vmem.set_value(self.Vmem_init)
        
class input_layer_const_rate(object):
    """ Input layer with constant injection current """
    def __init__(self, input, n_in, unit_input_step, threshold, initVmem=0.5, batchsize=1000):
        self.input = input # input real-valued matrix, shape = (batchsize, n_in)
        self.n_in = n_in
        self.unit_input_step = unit_input_step
        self.threshold = threshold
        self.signs = T.sgn(self.input)
        self.input_abs = abs(self.input)
        self.incremental_step = self.input_abs * self.unit_input_step
        self.initVmem = initVmem
        self.batchsize = batchsize
        self.Vmem_init = np.zeros((self.batchsize, n_in), dtype=theano.config.floatX) + np.float32(self.initVmem * threshold)
        self.Vmem = theano.shared(self.Vmem_init, name='Vmem', borrow=True)
        self.Vmem_update_prespike = self.Vmem + self.incremental_step
        self.output = T.ge(self.Vmem_update_prespike, self.threshold) * self.signs
        self.Vmem_update_postspike = T.cast(self.Vmem_update_prespike - abs(self.output) * self.threshold, theano.config.floatX)
    def refresh(self):
        self.Vmem.set_value(self.Vmem_init)
        
class input_layer(object):
    """ Input layer with probabilistic outputs, input is in the range of [0, 1] """
    def __init__(self, input, n_in, numStepsPerSample, batchsize=1000, deterministic=False):
        self.input = input
        self.n_in = n_in
        self.batchsize = batchsize
        self.deterministic = deterministic
        self.rng_mrg = RandomStreams(np.random.randint(1, 23943493))
        if self.deterministic == False:
            self.output = T.gt(input, self.rng_mrg.uniform((batchsize, n_in), low=0.0, high=1.0, dtype=theano.config.floatX))
        else:
            self.output = T.round(input)
            # self.output = T.cast(T.switch(T.ge(input, 0.3), 1.0, 0.), theano.config.floatX)
    
    def refresh(self):
        pass
    
class input_layer_2d(object):
    """ Input layer (2D) with probabilistic outputs, input is in the range of [0, 1] """
    def __init__(self, input, input_shape, numStepsPerSample, batchsize=1000, deterministic=False, binary=True, n_bits=2):
        self.input = input
        self.input_shape = input_shape
        self.output_shape = self.input_shape
        self.batchsize = batchsize
        self.deterministic = deterministic
        self.rng_mrg = RandomStreams(np.random.randint(1, 23943493))
        if self.deterministic == False:
            self.output = T.cast(T.gt(input, self.rng_mrg.uniform(self.input_shape, low=0.0, high=1.0, dtype=theano.config.floatX)), theano.config.floatX)
        else:
            self.output = T.cast(T.round(input), theano.config.floatX)
            # self.output = T.cast(T.switch(T.ge(input, 0.3), 1.0, 0.), theano.config.floatX)
        if binary == False:
            if n_bits == -1: # no quantization at all
                self.output = input
            else:
                # Normalize to [0 ~ 1 - 2^(-n_bits)]
                input_normed = input * (1 - 2**(-n_bits))
                if deterministic == False:
                    input_ceil = T.ceil(input_normed * 2**n_bits) / 2**n_bits
                    input_floor = T.floor(input_normed * 2**n_bits) / 2**n_bits
                    input_above_floor = input - input_floor
                    self.output = T.cast(T.switch(T.ge(input_above_floor, self.rng_mrg.uniform(self.input_shape, low=0.0, high=2**(-n_bits), dtype=theano.config.floatX)), input_ceil, input_floor), theano.config.floatX)
                else:
                    self.output = T.cast(T.round(input_normed * 2**n_bits) / 2**n_bits, theano.config.floatX)
        else:
            if deterministic == False:
                self.output = T.cast(T.gt(input, self.rng_mrg.uniform(self.input_shape, low=0.0, high=1.0, dtype=theano.config.floatX)), theano.config.floatX)
            else:
                # input_var_deterministic_binarized = T.round(input_var)
                self.output = T.cast(T.round(input), theano.config.floatX)
    def refresh(self):
        pass
    
class SNN_BA_Config(object):
    """ Configuration for SNN """
    def __init__(self, snnsize, threshold=1.0, initVmem=0.5, numStepsPerSample=512, deterministic=False):
        self.snnsize = snnsize
        self.W = None
        self.batchsize = 1000
        self.threshold = threshold
        self.initVmem = initVmem
        self.numStepsPerSample = numStepsPerSample
        self.unit_bias_steps = None
        self.deterministic = deterministic
        self.output_decoding = 'spike-count'
        self.output_zero_threshold = True
        self.memoryless = True
        self.input_coding = 'Poisson_Rate' # other option: 'Const_Rate'
        self.input_shape = snnsize[0]
        self.border_mode = 'valid'
        self.pooling_list = [True] * (len(self.snnsize) - 1)
        self.input_binary = True
        self.input_n_bits = 2
        
class SNN_BA(object):
    """ Spiking Neural Network with temporal-coding input, Integrate-Fire neuron model, only for testing"""
    def __init__(self, cfg):
        self.layers = []
        self.W = cfg.W
        self.threshold = cfg.threshold
        self.batchsize = cfg.batchsize
        self.numStepsPerSample = cfg.numStepsPerSample
        self.snnsize = cfg.snnsize
        self.initVmem = cfg.initVmem
        self.unit_bias_steps = cfg.unit_bias_steps
        self.deterministic = cfg.deterministic
        self.numlayers = len(self.snnsize) - 1
        self.outputRaster = np.empty([0, self.snnsize[-1], self.numStepsPerSample], dtype='int8')
        self.spikeCount = np.empty([0, self.snnsize[-1]], dtype='int16')
        self.x = T.matrix('x')
        self.predictions = [];
        self.lastlayer = self.numlayers
        self.output_decoding = cfg.output_decoding
        self.output_zero_threshold = cfg.output_zero_threshold
        self.testErr = 0.
        self.memoryless = cfg.memoryless
        self.input_coding = cfg.input_coding
        
        input = self.x
        if self.input_coding == 'Poisson_Rate':
            layer_input = input_layer(input=input, n_in=self.snnsize[0], numStepsPerSample=self.numStepsPerSample, batchsize=self.batchsize, deterministic=self.deterministic)
        elif self.input_coding == 'Const_Rate':
            layer_input = input_layer_const_rate(input=input, n_in=self.snnsize[0], unit_input_step=1., 
                                  threshold=self.threshold, initVmem=self.initVmem, batchsize=self.batchsize)
        else:
            print('Input coding: %s is not supported!' % self.input_coding)
        self.layers.append(layer_input)
        for i in range(self.lastlayer):
            if i == self.lastlayer-1 and self.output_zero_threshold == True:
                threshold = 0.
            else:
                threshold = self.threshold            
            layer = mlp(input=self.layers[-1].output, n_in=self.snnsize[i], n_out=self.snnsize[i+1], W=self.W[i], 
                        unit_bias_step=self.unit_bias_steps[i], threshold=threshold, initVmem=self.initVmem, batchsize=self.batchsize)
            self.layers.append(layer)
            
    def build_feedforward_function(self, test_shared_x):
        # Build feedfoward function for given test dataset
        print("Building feedforward theano function ...")
        index = T.lscalar('index')
        updates = collections.OrderedDict()
        batchsize = self.batchsize
        for i in range(1, self.lastlayer+1):
            param = self.layers[i].Vmem
            updates[param] = self.layers[i].Vmem_update_postspike
        
        feedfoward_onestep_fn = theano.function(inputs=[index], outputs=self.layers[self.lastlayer].output,
                                                  updates=updates, givens={self.x: test_shared_x[index*batchsize: (index+1)*batchsize]})
        return feedfoward_onestep_fn
        
    def refresh(self):
        for i in range(self.lastlayer+1):
            self.layers[i].refresh()
            
    def sim(self, test_x, test_y):
        # Prepare test_x (padding zeros if not integer multiple of batchsize)
        #   test_x: numpy array, float32, shape:(numSamples, inputDim)
        #   test_y: numpy array, float32, shape:(numSamples,)
        print("Preparing test dataset ...")
        numSamples = test_x.shape[0]
        inputDim = test_x.shape[1]
        batchsize = self.batchsize
        remainSamples = numSamples % batchsize
        if remainSamples != 0:
            padding_zeros = np.zeros((batchsize-remainSamples, inputDim), dtype = theano.config.floatX)
            test_x = np.vstack((test_x, padding_zeros))
        test_shared_x = theano.shared(np.float32(test_x), name = 'x', borrow = True)
        
        # Build test function
        test_fn = self.build_feedforward_function(test_shared_x)
        
        # Iterate over all the batches
        numBatches = int(np.ceil(numSamples / batchsize))
        self.outputRaster = np.empty([0, self.snnsize[self.lastlayer], self.numStepsPerSample], dtype='int8')        
        for i in range(numBatches):
            start_time = time.time()
            print("Simulating batch %d / %d ..." % (i+1, numBatches))
            self.refresh()
            outputRaster = np.empty([batchsize, self.snnsize[self.lastlayer], 0], dtype='int8')
            for j in range(self.numStepsPerSample):
                outputSpikes = test_fn(i)
                if self.memoryless == True:
                    self.refresh()
                outputSpikes = np.expand_dims(outputSpikes, axis=2)
                outputRaster = np.concatenate((outputRaster, outputSpikes), axis=2)
                # import pdb; pdb.set_trace()
            self.outputRaster = np.concatenate((self.outputRaster, outputRaster), axis=0)
            print("    Elaplsed time: %f" % (time.time()-start_time))
            # import pdb; pdb.set_trace()
            
        # Remove padding zeros and report statistics
        if remainSamples != 0:
            self.outputRaster = np.delete(self.outputRaster, range(numSamples, numBatches * batchsize), axis=0)
        if self.lastlayer == self.numlayers:
            outputRasterExtended = np.concatenate((self.outputRaster, np.ones((numSamples, self.snnsize[self.lastlayer], 1), dtype='int8')), axis=2)
            if self.output_decoding == 'first-to-spike':
                First_to_Spike = np.zeros((numSamples, self.snnsize[-1]))
                for i in range(numSamples):
                    for j in range(self.snnsize[-1]):
                        fireornot = outputRasterExtended[i,j,:]
                        First_to_Spike[i, j] = np.where(fireornot==1)[0][0]
                self.predictions = np.argmin(First_to_Spike, axis=1)
                
            elif self.output_decoding == 'spike-count':
                spikeCount = np.sum(outputRasterExtended, axis=2)
                self.predictions = np.argmax(spikeCount, axis=1)
                
            else: #last-step
                lastSpike = outputRasterExtended[:,:,-1]
                self.predictions = np.argmax(lastSpike, axis=1)
            self.testErr = np.int32(np.not_equal(self.predictions, test_y)).sum()/numSamples
            print("Test Error is %.3f%%" % (self.testErr * 100))
        # import pdb; pdb.set_trace()
                
                
class ConvSNN_BA(object):
    """ Convolutional Spiking Neural Network with temporal-coding input, Integrate-Fire neuron model, only for testing"""
    def __init__(self, cfg):
        self.layers = []
        self.W = cfg.W
        self.threshold = cfg.threshold
        self.batchsize = cfg.batchsize
        self.numStepsPerSample = cfg.numStepsPerSample
        self.snnsize = cfg.snnsize
        self.initVmem = cfg.initVmem
        self.unit_bias_steps = cfg.unit_bias_steps
        self.deterministic = cfg.deterministic
        self.numlayers = len(self.snnsize)
        self.outputRaster = np.empty([0, self.snnsize[-1], self.numStepsPerSample], dtype='int8')
        self.spikeCount = np.empty([0, self.snnsize[-1]], dtype='int16')
        self.x = T.tensor4('x')
        self.predictions = [];
        self.lastlayer = self.numlayers
        self.output_decoding = cfg.output_decoding
        self.output_zero_threshold = cfg.output_zero_threshold
        self.testErr = 0.
        self.input_shape = cfg.input_shape
        self.border_mode = cfg.border_mode
        self.pooling_list = cfg.pooling_list
        self.input_binary = cfg.input_binary
        self.input_n_bits = cfg.input_n_bits
        
        input = self.x
        layer_input = input_layer_2d(input=input, input_shape = (self.batchsize, self.input_shape[1], self.input_shape[2], self.input_shape[3]), numStepsPerSample=self.numStepsPerSample, batchsize=self.batchsize, deterministic=self.deterministic, binary=self.input_binary, n_bits=self.input_n_bits)
        self.layers.append(layer_input)
        conv_ix = 0
        for i in range(self.lastlayer):
            if i == self.lastlayer-1 and self.output_zero_threshold == True:
                threshold = 0.
            else:
                threshold = self.threshold   
            if len(self.W[i*2].shape) == 4: # Conv layer
                layer = conv(input=self.layers[-1].output, input_shape=self.layers[-1].output_shape, W=self.W[i*2], b=self.W[i*2+1], 
                        unit_bias_step=self.unit_bias_steps[i], threshold=threshold, initVmem=self.initVmem, batchsize=self.batchsize, border_mode=self.border_mode, maxpooling = self.pooling_list[conv_ix])
                conv_ix += 1
            else: # MLP layer
                layer = mlp(input=self.layers[-1].output, n_in=np.prod(self.layers[-1].output_shape[1:]), n_out=self.snnsize[i], W=np.vstack((self.W[2*i+1].reshape(1,-1), self.W[2*i])), 
                        unit_bias_step=self.unit_bias_steps[i], threshold=threshold, initVmem=self.initVmem, batchsize=self.batchsize)
            self.layers.append(layer)
            
    def build_feedforward_function(self, test_shared_x):
        # Build feedfoward function for given test dataset
        print("Building feedforward theano function ...")
        index = T.lscalar('index')
        updates = collections.OrderedDict()
        batchsize = self.batchsize
        for i in range(1, self.lastlayer+1):
            param = self.layers[i].Vmem
            updates[param] = self.layers[i].Vmem_update_postspike
        
        feedfoward_onestep_fn = theano.function(inputs=[index], outputs=self.layers[self.lastlayer].output,
                                                  updates=updates, givens={self.x: test_shared_x[index*batchsize: (index+1)*batchsize]})
        return feedfoward_onestep_fn
        
    def refresh(self):
        for i in range(self.lastlayer+1):
            self.layers[i].refresh()
            
    def sim(self, test_x, test_y):
        # Prepare test_x (padding zeros if not integer multiple of batchsize)
        #   test_x: numpy array, float32, shape:(numSamples, inputDim)
        #   test_x: numpy array, float32, shape:(numSamples, inputDim)
        print("Preparing test dataset ...")
        numSamples = test_x.shape[0]
        inputDim = test_x.shape[1:]
        batchsize = self.batchsize
        remainSamples = numSamples % batchsize
        if remainSamples != 0:
            padding_zeros = np.zeros((batchsize-remainSamples, inputDim), dtype = theano.config.floatX)
            test_x = np.vstack((test_x, padding_zeros))
        test_shared_x = theano.shared(np.float32(test_x), name = 'x', borrow = True)
        
        # Build test function
        test_fn = self.build_feedforward_function(test_shared_x)
        
        # Iterate over all the batches
        numBatches = int(np.ceil(numSamples / batchsize))
        self.outputRaster = np.empty([0, self.snnsize[self.lastlayer-1], self.numStepsPerSample], dtype='int8')        
        for i in range(numBatches):
            start_time = time.time()
            print("Simulating batch %d / %d ..." % (i+1, numBatches))
            self.refresh()
            outputRaster = np.empty([batchsize, self.snnsize[self.lastlayer-1], 0], dtype='int8')
            for j in range(self.numStepsPerSample):
                outputSpikes = test_fn(i)
                self.refresh()
                outputSpikes = np.expand_dims(outputSpikes, axis=2)
                outputRaster = np.concatenate((outputRaster, outputSpikes), axis=2)
                # import pdb; pdb.set_trace()
            self.outputRaster = np.concatenate((self.outputRaster, outputRaster), axis=0)
            print("    Elaplsed time: %f" % (time.time()-start_time))
            # import pdb; pdb.set_trace()
            
        # Remove padding zeros and report statistics
        if remainSamples != 0:
            self.outputRaster = np.delete(self.outputRaster, range(numSamples, numBatches * batchsize), axis=0)
        if self.lastlayer == self.numlayers:
            outputRasterExtended = np.concatenate((self.outputRaster, np.ones((numSamples, self.snnsize[self.lastlayer-1], 1), dtype='int8')), axis=2)
            if self.output_decoding == 'first-to-spike':
                First_to_Spike = np.zeros((numSamples, self.snnsize[-1]))
                for i in range(numSamples):
                    for j in range(self.snnsize[-1]):
                        fireornot = outputRasterExtended[i,j,:]
                        First_to_Spike[i, j] = np.where(fireornot==1)[0][0]
                self.predictions = np.argmin(First_to_Spike, axis=1)
                
            elif self.output_decoding == 'spike-count':
                spikeCount = np.sum(outputRasterExtended, axis=2)
                self.predictions = np.argmax(spikeCount, axis=1)
                
            else: #last-step
                lastSpike = outputRasterExtended[:,:,-1]
                self.predictions = np.argmax(lastSpike, axis=1)
                        
            self.testErr = np.int32(np.not_equal(self.predictions, test_y)).sum()/numSamples
            print("Test Error is %.3f%%" % (self.testErr * 100))        