# Standard Imports
import os
import sys
import math
import time
from datetime import datetime
from inspect import currentframe, getframeinfo

import contextlib

# Numpy & Scipy Imports
import numpy as np
import matplotlib.pyplot as plt

## Tensorflow Imports ##
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import keras

## Local Imports ##
from proj_utils import save_meta, load_meta
from metrics import NormRootMeanSquaredError , SoftmaxCategoricalCrossEntropyLoss
from hopf_layers import HopfRNNLayerTheta
from optimizers import MyOptimizer
from initializers import GlorotNorm , Eye

from lmu.lmu_layers import LMU

@contextlib.contextmanager
def options( new_ops ):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options( new_ops )
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options( old_opts )


''' Records the epoch training times and places them in the model's history. '''
class TrackTimeCallback( tf.keras.callbacks.Callback ):
    
    def __init__( self ):
        self.start = 0.0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time_ns()

    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = ( time.time_ns() - self.start ) / 1e9  # Time in seconds



''' Generates a dictionary object that is a wrapper  for a model.
    It contains useful stuff used, along with the model itself, 
    during the course of training and testing. Information is added to and 
    removed from the structure periodically during the course of training
    and testing. In the end this structure is saved as a json file in
    connection with the model. '''
def create_struct( model_name , train_dir ):
    
    # The itteration of training (Updated after we check if a meta.json exists)
    gen = 0
    
    model_struct = {}
    model_struct['name'] = model_name
    
    # Model directory with for just name
    model_dir = os.path.join( train_dir , model_name )
    model_struct['dir'] = model_dir
    
    # Load meta will return an empty dictionary if file doesn't exist
    model_meta = load_meta( model_dir )
    
    # Collect meta information for training
    if len( model_meta ) > 0:
        gen = model_meta['gen'] + 1
        
    model_struct['gdir'] = os.path.join( model_dir , 'gen_'+str(gen) )
    
    model_struct['chkpnt_dir'] = {}
    
    # Update meta information
    model_meta['gen'] = gen
    if 'csvset' not in model_meta: model_meta['csvset'] = 0
    model_struct['meta'] = model_meta
        
    return model_struct
    
''' Finalizes the dictionary object discussed in the create_struct() function above. '''
def finalize_struct(
                    model_struct,
                    model,
                    loss = None,
                    loss_monitor='val_loss',
                    loss_mode='val_min',
                    optimizer = None,
                    metrics=[],
                    chkpnts=[],
                    shuffle=False,
                    verbose=0
                 ):
    
    model_struct['chkpnt_dir'] = model_struct['gdir'] + '\\training\\checkpoints'
    model_struct['chkpnt_dir2'] = model_struct['gdir'] + '\\training\\checkpoints'
    
    model_struct['shuffle'] = shuffle
    
    if optimizer is None:
        optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    if loss is None:
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.AUTO,
            name='mse'
        )
        
    # Always add the standard loss callback.
    loss_chkpnt = tf.keras.callbacks.ModelCheckpoint(
        filepath = model_struct['chkpnt_dir'] + '\\cp-{epoch:04d}.ckpt',
        monitor = loss_monitor,
        verbose = verbose,
        mode = loss_mode,
        save_weights_only = True,
        save_best_only = True
    )
    
    
    # Setup traing and evaluation callbacks
    callbacks=[]
    callbacks.append( loss_chkpnt )
    for chkpnt in chkpnts:
        callbacks.append( chkpnt )
    callbacks.append( TrackTimeCallback() )
    model_struct['callbacks'] = callbacks
    
    # Add a standard time-series forcasting metric.
    if metrics is not None:
        if len( metrics ) == 0:
            metrics = [ tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' ) ]
            
    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = metrics
    )
    model.summary(
        line_length = None,
        positions = None,
        print_fn = print,
        expand_nested = True,
        show_trainable = True,
        layer_range = None
    )
    model_struct['model'] = model
    
    # Save updated meta.json information
    model_meta = model_struct['meta']
    save_meta( model_meta , model_struct['dir'] )
    del model_struct['meta']
    
    ## Get parameters and add for building excel data ##
    params_dict = {}
    sumstr = []
    model_struct['model'].summary(print_fn=lambda x: sumstr.append( x ) , expand_nested=True , show_trainable=True )
    ds = [ sumstr[2].find('Output') , sumstr[2].find('Param') , sumstr[2].find('Trainable') ]
    dash = False
    for ss in sumstr[4::]:
        if '__________' in ss: break
        if '=======' in ss:
            dash = True
            continue
        if not dash:
            pram_name = ss[0:ds[0]].split('(')[0].replace(' ','').lower().replace(',','')
            pram_value = ss[ds[1]:ds[2]].replace(' ','').replace(',','')
            trnble = ss[ds[2]::]
            if len( pram_name ) > 0:
                trnble = '_trnble' if 'Y' in ss[ds[2]::] else '_non-trnble'
                params_dict[ pram_name + '_params' ] = pram_value
        else:
            _ss = ss.split(' ')
            params_dict[ _ss[0].lower()+'_param' ] = _ss[-1].replace(' ','').replace(',','')
    
    model_struct['params'] = params_dict
    
    return model_struct
    
    
def mkygls_hopf_theta_callback( input_shape , output_shape , batch_size , train_dir , **model_spec ):
    
    name = model_spec['name'] if 'name' in model_spec else 'unknown'
    rwgt = model_spec['rec_wgt'] if 'rec_wgt' in model_spec else 'o'
    iwgt = model_spec['inpt_wgt'] if 'inpt_wgt' in model_spec else 'gu'
    act = model_spec['activation'] if 'activation' in model_spec else 'n'
    unts =  model_spec['units']  if 'units' in model_spec else 4
    
    test_name = name + '_'
    test_name += str( rwgt ) + '_' + str( iwgt ) + '_' 
    test_name += str( act ) + '_' + str( batch_size ) + '_'
    test_name += str( input_shape[0] ) + '_' + str( unts )
    print( 'Testing: ' , test_name , '\n' )
    
    _dtype = tf.complex64
    
    optims = {
        'layout_optimizer':True,
        'constant_folding':True,
        'shape_optimization':True,
        'remapping':True,
        'arithmetic_optimization':True,
        'dependency_optimization':False,
        'loop_optimization':False,
        'function_optimization':True,
        'debug_stripper':False,
        'disable_model_pruning':False,
        'scoped_allocator_optimization':False,
        'pin_to_host_optimization':False,
        'implementation_selector':True,
        'auto_mixed_precision':False,
        'disable_meta_optimizer':False
    }
    with options( optims ):
        new_ops = tf.config.optimizer.get_experimental_options()
        
        model_struct = create_struct( test_name , train_dir )
        input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
        hopf1 = HopfRNNLayerTheta(
            units = unts,
            activation = act,
            recurrent_weight = rwgt,
            input_weight = iwgt,
            return_sequences = True,
            stateful = True,
            train_weights = True,
            save_weights = True,
            dtype = _dtype,
            name = 'L1_hopf'
        )( input )
        dens1 = tf.keras.layers.Dense(
            output_shape[-1],
            kernel_initializer = tf.keras.initializers.Identity(),
            activation = 'relu',
            trainable = True
        )( hopf1 )
        model = tf.keras.Model( input , dens1 , name = 'mkygls_hopf_theta' )
    
    model_struct[ 'wgtlst' ] = [ 'A' , 'B' , 'dens' ]
    
    # Construct any other checkpoints or addons
    _metrics = [
        tf.keras.metrics.RootMeanSquaredError( name = 'rmse' ),
#        tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' )
    ]
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 1.e-3 )
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    _chkpnts = []
    
    _chkpnts.append(
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            verbose = 1,
            mode = 'val_min',
            baseline = None,
            restore_best_weights = False
        )
    )
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts
    )
    
    if 'params' in model_struct:
        model_struct['params']['units'] = str( unts )
        model_struct['params']['input_shape'] = str( input_shape ).replace( ',' , '' )
        model_struct['params']['output_shape'] = str( output_shape ).replace( ',' , '' )
    
    return model_struct


def cpymem_hopf_theta_callback( input_shape , output_shape , batch_size , train_dir , **model_spec ):
    
    name = model_spec['name'] if 'name' in model_spec else 'Unknown'
    rwgt = model_spec['rec_wgt'] if 'rec_wgt' in model_spec else 'O'
    iwgt = model_spec['inpt_wgt'] if 'inpt_wgt' in model_spec else 'GN'
    act = model_spec['activation'] if 'activation' in model_spec else 'H'
    unts =  model_spec['units']  if 'units' in model_spec else 4
    
    test_name = name + '_'
    test_name += str( rwgt ) + '_' + str( iwgt ) + '_' 
    test_name += str( act ) + '_' + str( batch_size ) + '_'
    test_name += str( input_shape[0] ) + '_' + str( unts )
    print( 'Testing: ' , test_name , '\n' )
    
    _dtype = tf.complex64
    
    ## Initializes the dense layer weights. Allows saving/loading of weight values. ##
    def dense_layer_weight_init( shape , dtype ):
        wgt = GlorotNorm( name = 'GN_u' + str( unts ) + '_dens' , save = True )
        return tf.math.real( wgt( shape , dtype ) )
    
    optims = {
        'layout_optimizer':True,
        'constant_folding':True,
        'shape_optimization':True,
        'remapping':True,
        'arithmetic_optimization':True,
        'dependency_optimization':False,
        'loop_optimization':False,
        'function_optimization':True,
        'debug_stripper':False,
        'disable_model_pruning':False,
        'scoped_allocator_optimization':False,
        'pin_to_host_optimization':False,
        'implementation_selector':True,
        'auto_mixed_precision':False,
        'disable_meta_optimizer':False
    }
    with options( optims ):
        new_ops = tf.config.optimizer.get_experimental_options()
        
        model_struct = create_struct( test_name , train_dir )
        input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
        hopf1 = HopfRNNLayerTheta(
            units = unts,
            output_size = output_shape[-1],
            activation = act,
            recurrent_weight = rwgt,
            input_weight = iwgt,
            return_sequences = True,
            stateful = True,
            train_weights = True,
            save_weights = True,
            dtype = _dtype,
            name = 'L1_hopf'
        )( input )
        dens1 = tf.keras.layers.Dense(
            output_shape[-1],
            kernel_initializer = tf.keras.initializers.Identity(),
            activation = 'relu',
            trainable = True
        )( hopf1 )
        model = tf.keras.Model( input , dens1 , name = 'cpymem_hopf_theta' )
    
    model_struct[ 'wgtlst' ] = [ 'A' , 'B' , 'dens' ]
    
    _loss = SoftmaxCategoricalCrossEntropyLoss( name = 'soft_CCE' )
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        verbose = 1,
        mode = 'val_min',
        baseline = None,
        restore_best_weights = False
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts,
        shuffle = False
    )
    
    return model_struct


def cpymem_gru_callback( input_shape , output_shape , batch_size , train_dir , **model_spec ):
    
    name = model_spec['name'] if 'name' in model_spec else 'Unknown'
    rwgt = model_spec['rec_wgt'] if 'rec_wgt' in model_spec else 'O'
    iwgt = model_spec['inpt_wgt'] if 'inpt_wgt' in model_spec else 'GN'
    act = model_spec['activation'] if 'activation' in model_spec else 'H'
    unts =  model_spec['units']  if 'units' in model_spec else 4
    
    model_struct = create_struct( 'gru' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    gru1 = tf.keras.layers.GRU(
        units = unts,
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        kernel_initializer = 'glorot_uniform',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        name = 'L1_gru'
    )( input )
    #'''
    gru2 = tf.keras.layers.GRU(
        units = unts,
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        kernel_initializer = 'glorot_uniform',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        name = 'L2_gru'
    )( gru1 )
    dens1 = tf.keras.layers.Dense( output_shape[-1] , activation = 'relu' )( gru2 )
    model = tf.keras.Model( input , dens1 , name = 'cpymem_gru' )
    '''
    dens1 = tf.keras.layers.Dense( output_shape[-1] , activation = 'relu' )( gru1 )
    model = tf.keras.Model( input , dens1 , name = 'cpymem_gru' )
    #'''
    
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' , 'bias' ]
    
    _loss = SoftmaxCategoricalCrossEntropyLoss( name = 'soft_CCE' )
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None # [ tf.keras.metrics.CategoricalCrossentropy( name = 'catcrossent' ) ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 25,
        verbose = 1,
        mode = 'min',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct