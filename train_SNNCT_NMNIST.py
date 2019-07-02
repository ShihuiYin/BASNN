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

# Description: Test BASNN on n-mnist
# Created on 04/27/2017
# Modified on 05/02/2017, add the opposite saccade direction


import numpy as np
import h5py
np.random.seed(1234) # for reproducibility?

import lasagne
import theano
import theano.tensor as T

# from pylearn2.datasets.mnist import MNIST
from collections import OrderedDict
from train import train_sgd
import BASNN
import sys
sys.setrecursionlimit(50000)

if __name__ == "__main__":
    
    batch_size = 100
    LR_start = 0.001 #0.1
    LR_finish = 0.000001
    num_epochs = 100  
    LR_decay = (LR_finish/LR_start)**(1./num_epochs)          
    dropout_in = 0.2
    dropout_hidden = 0.2
    n_hidden_layers = 2
    num_units = 1024
    separation = 1.0
    num_time_steps = int(sys.argv[1])
    shuffle_parts = 1
    
    # load dataset
    print('Loading N-MNIST dataset (time steps = %d) ...' % num_time_steps)    
    train_set = h5py.File('./data/NMNIST-Train_%d.mat' % num_time_steps)
    train_set_x = train_set['Data']
    train_set_x = train_set_x[()]
    train_set_x = np.swapaxes(train_set_x, 0, 2)
    train_set_y = train_set['labels']
    train_set_y = train_set_y[()].transpose()
    
    test_set = h5py.File('./data/NMNIST-Test_%d.mat' % num_time_steps)
    test_set_x = test_set['Data']
    test_set_x = test_set_x[()]
    test_set_x = np.swapaxes(test_set_x, 0, 2).astype(theano.config.floatX)
    test_set_y = test_set['labels']
    test_set_y = test_set_y[()].transpose().astype('int8')
    
    valid_set_x = train_set_x[:,50000:60000,:].astype(theano.config.floatX)
    train_set_x = train_set_x[:,0:50000,:].astype(theano.config.floatX)
    valid_set_y = train_set_y[50000:60000].astype('int8')
    train_set_y = train_set_y[0:50000].astype('int8')
    
    # flatten targets
    train_set_y = np.hstack(train_set_y)
    valid_set_y = np.hstack(valid_set_y)
    test_set_y = np.hstack(test_set_y)
    
    # Onehot the targets
    train_set_y = np.float32(np.eye(10)[train_set_y])    
    valid_set_y = np.float32(np.eye(10)[valid_set_y])
    test_set_y = np.float32(np.eye(10)[test_set_y])
    
    # for hinge loss    
    train_set_y = np.stack([2. * separation * train_set_y - separation] * num_time_steps)
    valid_set_y = np.stack([2. * separation * valid_set_y - separation] * num_time_steps)
    test_set_y = np.stack([2. * separation * test_set_y - separation] * num_time_steps)
    
    # add the opposite saccade direction
    train_set_x_new = np.flipud(train_set_x)
    train_set_x = np.concatenate((train_set_x, train_set_x_new), axis=1)
    train_set_y_1 = np.concatenate((train_set_y, np.ones((num_time_steps, 50000, 1), dtype='float32')), axis=2)
    train_set_y_2 = np.concatenate((train_set_y, -np.ones((num_time_steps, 50000, 1), dtype='float32')), axis=2)
    train_set_y = np.concatenate((train_set_y_1, train_set_y_2), axis=1)
    
    valid_set_x_new = np.flipud(valid_set_x)
    valid_set_x = np.concatenate((valid_set_x, valid_set_x_new), axis=1)
    valid_set_y_1 = np.concatenate((valid_set_y, np.ones((num_time_steps, 10000, 1), dtype='float32')), axis=2)
    valid_set_y_2 = np.concatenate((valid_set_y, -np.ones((num_time_steps, 10000, 1), dtype='float32')), axis=2)
    valid_set_y = np.concatenate((valid_set_y_1, valid_set_y_2), axis=1)
    
    test_set_x_new = np.flipud(test_set_x)
    test_set_x = np.concatenate((test_set_x, test_set_x_new), axis=1)
    test_set_y_1 = np.concatenate((test_set_y, np.ones((num_time_steps, 10000, 1), dtype='float32')), axis=2)
    test_set_y_2 = np.concatenate((test_set_y, -np.ones((num_time_steps, 10000, 1), dtype='float32')), axis=2)
    test_set_y = np.concatenate((test_set_y_1, test_set_y_2), axis=1)
    # import pdb; pdb.set_trace()
    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor3('inputs')
    target = T.tensor3('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = BASNN.InputLayer(
            shape=(num_time_steps, batch_size, 1156),
            input_var=input,
            deterministic=True)
    
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
            
    for k in range(n_hidden_layers):

        mlp = lasagne.layers.DenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units,
                num_leading_axes=2)    # The first axis is num_time_steps, the 2nd axis is batch_size
                    
        mlp = BASNN.NonlinearityLayer(mlp, num_time_steps=num_time_steps)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
                
    mlp = lasagne.layers.DenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=11,
                num_leading_axes=2)    
    
    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,separation**2-target*train_output)))
    
    params = lasagne.layers.get_all_params(mlp, trainable=True)
    updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,separation**2-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=2), T.argmax(target, axis=2)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    train_sgd(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set_x,train_set_y,
            valid_set_x,valid_set_y,
            test_set_x,test_set_y,
            shuffle_parts=shuffle_parts,
            batch_size_axis=1,
            save_path="nmnist_param_temporal_coding_%d.npz" % num_time_steps,
            monitor_valid_err=0)
