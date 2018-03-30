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

# Description: Test BASNN on mnist (CNN)

import numpy as np
np.random.seed(1234) # for reproducibility

import lasagne
import theano
import theano.tensor as T

from pylearn2.datasets.mnist import MNIST
from collections import OrderedDict
import BASNN
import argparse
from ast import literal_eval as bool

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="BASNN CNN for MNIST", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nu', dest='num_units', type=int, default=512, help="Number of units in each hidden layer")
    parser.add_argument('-nl', dest='n_hidden_layers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('-n1', dest='n_conv_1', type=int, default=12, help="Number of channels in conv 1 layer")
    parser.add_argument('-n2', dest='n_conv_2', type=int, default=64, help="Number of channels in conv 2 layer")
    parser.add_argument('-di', dest='dropout_in', type=float, default=0.1, help="Dropout ratio for input layer")
    parser.add_argument('-dh', dest='dropout_hidden', type=float, default=0.3, help="Dropout ratio for hidden layers")
    parser.add_argument('-ne', dest='num_epochs', type=int, default=200, help="Number of epochs")
    parser.add_argument('-ls', dest='LR_start', type=float, default=0.001, help="Start learning rate")
    parser.add_argument('-lf', dest='LR_finish', type=float, default=0.00001, help="End learning rate")
    parser.add_argument('-la', dest='lambda_act', type=float, default=0., help="Regularization lambda for activations")
    parser.add_argument('-de', dest='deterministic', type=bool, default=False, help="If true, input is deterministic")
    parser.add_argument('-l1', dest='lambda_l1', type=float, default=0., help="L1 regularization")
    parser.add_argument('-l2', dest='lambda_l2', type=float, default=0., help="L2 regularization")
    parser.add_argument('-bs', dest='batch_size', type=int, default=100, help="Batch size")
    parser.add_argument('-sp', dest='save_path', default="MNIST_param_cnn.npz", help="Path to save parameters")
    args = parser.parse_args()
    print(args)
        
    num_units = args.num_units
    n_hidden_layers = args.n_hidden_layers
    n_conv_1 = args.n_conv_1
    n_conv_2 = args.n_conv_2
    dropout_in = args.dropout_in
    dropout_hidden = args.dropout_hidden
    num_epochs = args.num_epochs
    LR_start = args.LR_start
    LR_finish = args.LR_finish
    lambda_act = args.lambda_act
    lambda_l1 = args.lambda_l1
    lambda_l2 = args.lambda_l2
    batch_size = args.batch_size
    deterministic = args.deterministic
    save_path = args.save_path
    LR_decay = (LR_finish/LR_start)**(1./num_epochs)        
    shuffle_parts = 1    
    print("n_conv_1 = %d" % n_conv_1)
    print("n_conv_2 = %d" % n_conv_2)
    print("num_units = %d" % num_units)
    print("n_hidden_layers = %d" % n_hidden_layers)
    print("dropout_in = %.1f" % dropout_in)
    print("dropout_hidden = %.1f" % dropout_hidden)
    print("num_epochs = %d" % num_epochs)
    print("LR_start = %.0e" % LR_start)
    print("LR_finish = %.0e" % LR_finish)
            
    print('Loading MNIST dataset...')
    
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
    test_set = MNIST(which_set= 'test', center = False)
    
    train_set.X = train_set.X.reshape(-1, 1, 28, 28)
    valid_set.X = valid_set.X.reshape(-1, 1, 28, 28)
    test_set.X = test_set.X.reshape(-1, 1, 28, 28)
    
    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    loss = 0
    temp_sum = 0

    cnn = BASNN.InputLayer(
            shape=(batch_size, 1, 28, 28),
            input_var=input,
            deterministic=deterministic,
            binary=True,
            threshold=0.5,
            batch_size=batch_size)
    
    cnn = lasagne.layers.Conv2DLayer(
            cnn, num_filters=n_conv_1, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.HeUniform())
            
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
            
    cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=BASNN.hard_fire)
                
    cnn = lasagne.layers.Conv2DLayer(
            cnn, num_filters=n_conv_2, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.HeUniform())
            
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
            
    cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=BASNN.hard_fire)
            
    for k in range(n_hidden_layers):

        cnn = lasagne.layers.DenseLayer(
                cnn, 
                nonlinearity=lasagne.nonlinearities.identity,
                W=lasagne.init.HeUniform(),
                num_units=num_units)    
                
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=BASNN.hard_fire)
        
        if lambda_act > 0:
            layer_output = lasagne.layers.get_output(cnn, deterministic=True)
            if k < n_hidden_layers - 1:
                loss += T.mean(T.sqr(layer_output)) * num_units
                temp_sum += num_units
            else:
                loss += T.mean(T.sqr(layer_output)) * 10
                temp_sum += 10
        
        cnn = lasagne.layers.DropoutLayer(
                cnn, 
                p=dropout_hidden)
                
    cnn = lasagne.layers.DenseLayer(
                cnn, 
                nonlinearity=lasagne.nonlinearities.identity,
                W=lasagne.init.HeUniform(),
                num_units=10)   

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    if lambda_act > 0:
        loss *= (lambda_act / temp_sum)
    loss += T.mean(T.sqr(T.maximum(0.,1.-target*(train_output-1.))))
    
    params = lasagne.layers.get_all_params(cnn, trainable=True)
    if lambda_l2 > 0.:
        for param in params:
            if param.name == 'W':
                loss += T.sum(T.sqr(param)) * lambda_l2
    if lambda_l1 > 0.:
        for param in params:
            if param.name == 'W':
                loss += T.sum(abs(param)) * lambda_l1
    updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*(test_output-1.))))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    print("Parameters saved in %s" % save_path)
    BASNN.train_sgd(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y,
            shuffle_parts=shuffle_parts,
            save_path=save_path)
            
    print("Parameters saved in %s" % save_path)
