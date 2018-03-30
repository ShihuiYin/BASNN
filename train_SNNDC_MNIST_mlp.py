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

# Description: Test BASNN on mnist

import numpy as np
np.random.seed(1234) # for reproducibility

import lasagne
import theano
import theano.tensor as T

from pylearn2.datasets.mnist import MNIST
import sys
import BASNN
import argparse
from ast import literal_eval as bool

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="BASNN MLP for MNIST", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nu', dest='num_units', type=int, default=1024, help="Number of units in each hidden layer")
    parser.add_argument('-nl', dest='n_hidden_layers', type=int, default=2, help="Number of hidden layers")
    parser.add_argument('-di', dest='dropout_in', type=float, default=0.1, help="Dropout ratio for input layer")
    parser.add_argument('-dh', dest='dropout_hidden', type=float, default=0.3, help="Dropout ratio for hidden layers")
    parser.add_argument('-ne', dest='num_epochs', type=int, default=400, help="Number of epochs")
    parser.add_argument('-ls', dest='LR_start', type=float, default=0.001, help="Start learning rate")
    parser.add_argument('-lf', dest='LR_finish', type=float, default=0.00001, help="End learning rate")
    parser.add_argument('-la', dest='lambda_act', type=float, default=0., help="Regularization lambda for activations")
    parser.add_argument('-de', dest='deterministic', type=bool, default=False, help="If true, input is deterministic")
    parser.add_argument('-l1', dest='lambda_l1', type=float, default=0., help="L1 regularization")
    parser.add_argument('-l2', dest='lambda_l2', type=float, default=0., help="L2 regularization")
    parser.add_argument('-bs', dest='batch_size', type=int, default=100, help="Batch size")
    parser.add_argument('-sp', dest='save_path', default="MNIST_param_mlp.npz", help="Path to save parameters")
    args = parser.parse_args()
    print(args)
    
    num_units = args.num_units
    n_hidden_layers = args.n_hidden_layers
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
    
    print("num_units = %d" % num_units)
    print("n_hidden_layers = %d" % n_hidden_layers)
    print("dropout_in = %.1f" % dropout_in)
    print("dropout_hidden = %.1f" % dropout_hidden)
    print("num_epochs = %d" % num_epochs)
    print("LR_start = %.0e" % LR_start)
    print("LR_finish = %.0e" % LR_finish)
    print("lambda_act = %.0e" % lambda_act)
            
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

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    loss = 0
    temp_sum = 0

    mlp = BASNN.InputLayer(
            shape=(batch_size, 1, 28, 28),
            input_var=input,
            deterministic=deterministic,
            binary=True,
            threshold=0.5)
    
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
            
    for k in range(n_hidden_layers):

        mlp = lasagne.layers.DenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                W=lasagne.init.HeUniform(),
                num_units=num_units)    
                
        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=BASNN.hard_fire)
        
        if lambda_act > 0:
            layer_output = lasagne.layers.get_output(mlp, deterministic=True)
            if k < n_hidden_layers - 1:
                loss += T.mean(T.sqr(layer_output)) * num_units
                temp_sum += num_units
            else:
                loss += T.mean(T.sqr(layer_output)) * 10
                temp_sum += 10
        
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
                
    mlp = lasagne.layers.DenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                W=lasagne.init.HeUniform(),
                num_units=10)   

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # loss function
    if lambda_act > 0:
        loss *= (lambda_act / temp_sum)
    loss += T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    params = lasagne.layers.get_all_params(mlp, trainable=True)
    if lambda_l2 > 0.:
        for param in params:
            if param.name == 'W':
                loss += T.sum(T.sqr(param)) * lambda_l2
    if lambda_l1 > 0.:
        for param in params:
            if param.name == 'W':
                loss += T.sum(abs(param)) * lambda_l1
    updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
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
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y,
            shuffle_parts=shuffle_parts,
            save_path=save_path)
            
    print("Parameters saved in %s" % save_path)
