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

# Description: BASNN.py includes a few modules for Binary-Activation SNN

import lasagne
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import time


class HardFire(theano.Op):
    __props__ = ()
    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])
    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.greater(x, 1.).astype(theano.config.floatX)
    def grad(self, inputs, output_grads):
        return [0.5*output_grads[0] * (T.and_(T.gt(inputs[0], 0.), T.lt(inputs[0], 2.)))]

hard_fire = HardFire()

def hard_fire_testonly(x):
    return T.cast(T.gt(x, 1.,), theano.config.floatX)

class InputLayer(lasagne.layers.InputLayer):
    def __init__(self, shape, input_var=None, name=None, binary=True, deterministic=False, threshold=0.5, batch_size=100, n_bits=-1, **kwargs):
        self.rng_mrg = RandomStreams(lasagne.random.get_rng().randint(1, 2394349593))
        if binary == False:
            if n_bits == -1: # no quantization at all
                super(InputLayer, self).__init__(shape=shape, input_var=input_var, name=name, **kwargs)
            else:
                # Normalize to [0 ~ 1 - 2^(-n_bits)]
                input_var_normed = input_var * (1 - 2**(-n_bits))
                if deterministic == False:
                    shape_rand = list(shape)
                    if shape_rand[0] is None:
                        shape_rand[0] = batch_size
                    shape_rand = tuple(shape_rand)
                    input_var_ceil = T.ceil(input_var_normed * 2**n_bits) / 2**n_bits
                    input_var_floor = T.floor(input_var_normed * 2**n_bits) / 2**n_bits
                    input_var_above_floor = input_var - input_var_floor
                    input_var_stochastic_quantized = T.cast(T.switch(T.ge(input_var_above_floor, self.rng_mrg.uniform(shape_rand, low=0.0, high=2**(-n_bits), dtype=theano.config.floatX)), input_var_ceil, input_var_floor), theano.config.floatX)
                    super(InputLayer, self).__init__(shape=shape, input_var=input_var_stochastic_quantized, name=name, **kwargs)
                else:
                    input_var_deterministic_quantized = T.cast(T.round(input_var_normed * 2**n_bits) / 2**n_bits, theano.config.floatX)
                    super(InputLayer, self).__init__(shape=shape, input_var=input_var_deterministic_quantized, name=name, **kwargs)
        else:
            if deterministic == False:
                shape_rand = list(shape)
                if shape_rand[0] is None:
                    shape_rand[0] = batch_size
                shape_rand = tuple(shape_rand)
                # Bernoulli spikes
                input_var_stochastic_binarized = T.cast(T.gt(input_var, self.rng_mrg.uniform(shape_rand, low=0.0, high=1.0, dtype=theano.config.floatX)), theano.config.floatX)
                super(InputLayer, self).__init__(shape=shape, input_var=input_var_stochastic_binarized, name=name, **kwargs)
            else:
                input_var_deterministic_binarized = T.cast(T.switch(T.ge(input_var, threshold), 1.0, 0.), theano.config.floatX)
                super(InputLayer, self).__init__(shape=shape, input_var=input_var_deterministic_binarized, name=name, **kwargs)
            
class NonlinearityLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nonlinearity=hard_fire, num_time_steps=2, reset=True, **kwargs):
        super(NonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_time_steps = num_time_steps
        self.reset = reset
    def get_output_for(self, input, **kwargs):
        output_list = []
        output_list.append(self.nonlinearity(input[0]))
        for i in range(1,self.num_time_steps):
            temp = 0
            temp += input[i]
            for j in range(i):
                if self.reset == True:
                    temp += input[j] - output_list[j]
                else:
                    temp += input[j]
            output_list.append(self.nonlinearity(temp))
        output = T.stack(output_list)
        return output

def train_sgd(train_fn,val_fn,
            model,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test,
            save_path=None,
            shuffle_parts=1,
            batch_size_val=None,
            batch_size_axis=0,
            repeat_in_batch=1,
            monitor_valid_err=True,
            k_start=0,
            k_decay=1,
            save_last_epoch=False,
            k_decay_mode="exponential"
            ):
    
    if batch_size_val is None:
        batch_size_val = batch_size
    # A function which shuffles a dataset
    def shuffle(X,y):
        
        chunk_size = int(len(X)/shuffle_parts)
        shuffled_range = list(range(chunk_size))
        
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        
        for k in range(shuffle_parts):
            
            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer
        
        return X,y
            
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR,k=0):
        
        loss = 0
        
        if batch_size_axis == 0:
            batches = int(len(X)/batch_size)
            for i in range(batches):
                sample_indices = list(range(i*batch_size,(i+1)*batch_size))*repeat_in_batch
                if k > 0:
                    loss += train_fn(X[sample_indices],y[sample_indices],LR, k)
                else:
                    loss += train_fn(X[sample_indices],y[sample_indices],LR)
        elif batch_size_axis == 1:
            batches = int(X.shape[1]/batch_size)
            for i in range(batches):
                sample_indices = list(range(i*batch_size,(i+1)*batch_size))*repeat_in_batch
                if k > 0:
                    loss += train_fn(X[:,sample_indices],y[:,sample_indices],LR, k)
                else:
                    loss += train_fn(X[:,sample_indices],y[:,sample_indices],LR)
        else:
            print("Batch size axis = %d is not supported" % batch_size_axis)
        
        loss/=batches
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y,k=0):
        
        err = 0
        loss = 0
        
        if batch_size_axis == 0:
            batches = int(len(X)/batch_size_val)
            for i in range(batches):
                sample_indices = list(range(i*batch_size,(i+1)*batch_size))*repeat_in_batch
                if k > 0:
                    new_loss, new_err = val_fn(X[sample_indices], y[sample_indices], k)
                else:
                    new_loss, new_err = val_fn(X[sample_indices], y[sample_indices])
                err += new_err
                loss += new_loss
        elif batch_size_axis == 1:
            batches = int(X.shape[1]/batch_size)
            for i in range(batches):
                sample_indices = list(range(i*batch_size,(i+1)*batch_size))*repeat_in_batch
                if k > 0:
                    new_loss, new_err = val_fn(X[:,sample_indices], y[:,sample_indices], k)
                else:
                    new_loss, new_err = val_fn(X[:,sample_indices], y[:,sample_indices])
                err += new_err
                loss += new_loss
        else:
            print("Batch size axis = %d is not supported" % batch_size_axis)
        
        err = err / batches * 100
        loss /= batches

        return err, loss
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_val_loss = 100000
    best_epoch = 1
    LR = LR_start
    k = k_start
    k_finish = k_start * k_decay ** (num_epochs-1)
    k_increm = (k_finish - k_start) / (num_epochs-1)
    if k_start != 0:
        print("k_start = %f" % k_start)
    
    for epoch in range(num_epochs):
        
        start_time = time.time()
        if k > 0:
            train_loss = train_epoch(X_train,y_train,LR,k)
        else:
            train_loss = train_epoch(X_train,y_train,LR)
        X_train,y_train = shuffle(X_train,y_train)
        print('finish one epoch training')
        print(time.time() - start_time)
        if k > 0:
            val_err, val_loss = val_epoch(X_val,y_val,k=1)
        else:
            val_err, val_loss = val_epoch(X_val,y_val)
        print('finish one epoch validation')
        print(time.time() - start_time)
        
        if save_last_epoch == False:
            if monitor_valid_err == True:
                if val_err < best_val_err:
                    
                    best_val_err = val_err
                    best_epoch = epoch+1
                    if k > 0:
                        test_err, test_loss = val_epoch(X_test,y_test,k)
                    else:
                        test_err, test_loss = val_epoch(X_test,y_test)
                    if save_path is not None:
                        np.savez(save_path, *lasagne.layers.get_all_param_values(model))
            else: # monitor validation loss
                if val_loss < best_val_loss:
                    
                    best_val_loss = val_loss
                    best_epoch = epoch+1
                    
                    test_err, test_loss = val_epoch(X_test,y_test,k)
                    
                    if save_path is not None:
                        np.savez(save_path, *lasagne.layers.get_all_param_values(model))
        else:
            test_err, test_loss = val_epoch(X_test,y_test,k)
            np.savez(save_path, *lasagne.layers.get_all_param_values(model))
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        if k > 0:
            print("  k:                             "+str(k))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        if monitor_valid_err:
            print("  best validation loss:          "+str(val_loss))
            print("  best validation error rate:    "+str(best_val_err)+"%")  
        else:
            print("  best validation loss:          "+str(best_val_loss))
            print("  best validation error rate:    "+str(val_err)+"%")  
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        # decay the LR
        LR *= LR_decay
        if k_decay_mode == "exponential":
            k *= k_decay
        else:
            k += k_increm