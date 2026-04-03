''' Standard Imports '''
import os
import sys
import math
import random
from datetime import datetime
import time
import shutil



''' Special Imports '''
import numpy as np
np.set_printoptions( precision = 5 , threshold = 50 , edgeitems = 5 , linewidth = 150 , floatmode = 'fixed' )

## Prevents Tensorflow from accessing GPU ##
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Clear custom imports #
tf.keras.utils.get_custom_objects().clear()

''' Local Imports'''
import proj_utils as utils          # Print, Plot, & Helper functions 
from trainer import ModelTrainer    # Model trainer.

# Data Imports
from data import MackeyGlassGenerator , CopyMemoryGenerator

from models import cpymem_gru_callback

## Model imports for MY BASE RNN arch. ##
from models import mkygls_hopf_theta_callback , cpymem_hopf_theta_callback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )

def getint( s ):
    try:
        return int( s )
    except:
        return None

def getstr( s ):
    try:
        return str( s )
    except:
        return None

def build_spec( u , n , a , r , i , w ):
    modspc = {}
    modspc['units'] = u
    modspc['name'] = n
    modspc['activation'] = a
    modspc['rec_wgt'] = r
    modspc['inpt_wgt'] = i
    return modspc

def print_spec( **spec ):
    print( spec )
    spcstr = ''
    for k , v in spec.items(): spcstr += '  ' + k.capitalize() + ': ' + str( v )
    return spcstr[2::]  # Remove first two spaces

def mkygls_experiment( _input_size = 32 , _batch_size = 16 , _epoch_size = 10 , _num_epochs = 100 , **model_spec ):
    
    ''' ----- INPUT FEATURE PARAMETERS FOR MODEL ----- '''
    tf.keras.backend.clear_session()
    
    ## Allows best runtimes for training on largest sets ##
    tf.config.threading.set_inter_op_parallelism_threads( 25 )
#    tf.config.threading.set_inter_op_parallelism_threads( 32 )
    
    # Model params
    input_size = _input_size
    output_size = _input_size
    
    batch_size = _batch_size
    epoch_size = _epoch_size
    num_epochs = _num_epochs
    
    _time_steps = model_spec['units'] if 'units' in model_spec else 24
    
    step_size = output_size
    test_split = 0.25       # as a percent(%) of total dataset 
    valid_split = 0.25      # as a percent(%) of training set
    _verbose = 1
    
    # Dtype for data generation #
    data_dtype = np.float32
    wndo_str = str( input_size )
    
    ''' TRAIN MODEL '''
    trainer = ModelTrainer(
        data_callback = MackeyGlassGenerator,
        batch_size = batch_size,
        epoch_size = epoch_size,
        num_epochs = num_epochs,
        input_size = input_size,
        time_steps = _time_steps,
        output_size = output_size,
        step_size = step_size,
        test_split = test_split,
        valid_split = valid_split,
        use_dataset = True,
        dtype = data_dtype,
        name = 'mkygls' + '\\window_size_' + wndo_str
    )
    
    callback_list = [ mkygls_hopf_theta_callback ]
    
    start_time = time.time()
    
    ## Returns Successful Epoch Count Upon Failure ##
    ret = trainer.execute(
        model_callbacks = callback_list,
        fit_valid_split = valid_split,
        do_prediction = True,
        do_evaluation = True,
        verbose = _verbose,
        testing = None,
        load_chkpnt = None,
        safe_mode = True,
        **model_spec
    )
    
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)\n'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
    
    if ret < 0: return ret
    
    trainer.save(
        max_plot_len = 2048,
        show_plot = False,
        save_plot = True,
        show_weights = False,
        save_weights = True,
        verbose = _verbose,
        datnme = 'mkygls'
    )
    
    tf.keras.backend.clear_session()
    
    return ret
    
def cpymem_experiment( _input_size = 32 , _batch_size = 16 , _epoch_size = 10 , _num_epochs = 100 , **model_spec ):
    
    ''' ----- INPUT FEATURE PARAMETERS FOR MODEL ----- '''
    tf.keras.backend.clear_session()
    
    ## Allows sub-20's runtimes for training on largest sets ##
    tf.config.threading.set_inter_op_parallelism_threads( 4 )
    
    # Model params
    input_size = 10
    output_size = input_size - 2
    
    # Train params #
    batch_size = _batch_size
    epoch_size = _epoch_size
    num_epochs = _num_epochs
    
    step_size = 1
    test_split = 0.25       # as a percent(%) of training parameters
    valid_split = 0.25
    _verbose = 1
    
    # Dtype for data generation #
    data_dtype = np.float32
    
    seqstr = str( _input_size )
    
    ''' TRAIN MODEL '''
    trainer = ModelTrainer(
        data_callback = CopyMemoryGenerator,
        in_bits = input_size,
        out_bits = input_size-2,
        max_seq = _input_size,
        batch_size = batch_size,
        epoch_size = epoch_size,
        num_epochs = num_epochs,
        input_size = input_size,
        output_size = output_size,
        step_size = step_size,
        test_split = test_split,
        valid_split = valid_split,
        use_dataset = False,
        dtype = data_dtype,
        name = 'cpymem' + '\\seqlen_' + seqstr
    )
    
    callback_list = [ cpymem_hopf_theta_callback ]
    
    start_time = time.time()
    
    test_count = None
    
    ## Returns Successful Epoch Count Upon Failure ##
    ret = trainer.execute(
        model_callbacks = callback_list,
        fit_valid_split = valid_split,
        do_prediction = True,           # True , False
        do_evaluation = True,
        verbose = _verbose,
        testing = test_count,
        load_chkpnt = None,
        safe_mode = True,
        **model_spec
    )
    if test_count is not None: exit(0)
    
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)\n'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
    
    if ret < 0: return ret
    
    # Save all runs for callback_list models  -  True False
    trainer.save(
        max_plot_len = 2048,
        show_plot = False,
        save_plot = True,
        show_weights = False,
        save_weights = True,
        verbose = _verbose,
        datnme = 'cpymem'
    )
    
    tf.keras.backend.clear_session()
    
    return ret
    
def remove_weights( unts ):
    ## Delete the weights given as arguments ##
    for dir in os.listdir():
        if ( '_u' + str( unts ) + '_L1' in dir ) or ( '_u' + str( unts ) + '_dens.npy' in dir ):
            os.remove( dir )
    
def run_experiment_logic(
                          _experiment_func,
                          _exp_name = 'Unknown',
                          test_size = 1,
                          input_size = 32,
                          batch_size = 50,
                          epoch_size = 10,
                          num_epochs = 100,
                          _testing = False,
                          **spec
                        ):
    
    attempts , max_attempts , ret_check = 0 , 10 , 10    # attemps <int> counter , max_attemps <int> is number of failures #
    count_completed = 0
    
    if 'units' in spec: unts = spec['units']
    
    if not _testing:
    
        if num_epochs < ret_check: ret_check = num_epochs
        
        while count_completed < test_size:
            
            pstr = '\n' + '- '*50 + '\n'
            pstr += 'Running :: ' + _exp_name
            pstr += ' | Completed Runs ' + str( count_completed )
            pstr += ' | Failed Attempts ' + str( attempts )
            print( '\n' + pstr + '\n' )
            
            if attempts == max_attempts:
                pstr = 'Maximum Attempts Reached:' + str( max_attempts ) + '\n'
                pstr += 'Exiting early ... @\n'
                print( '\n' + pstr + '\n' )
                break
            
            ret = _experiment_func(
                _input_size = input_size,
                _batch_size = batch_size,
                _epoch_size = epoch_size,
                _num_epochs = num_epochs,
                **spec
            )
            
            ## Return value < 0 => failed training or prediction ... create new weight matrix and try again ##
            if ret < 0:
                attempts += 1
                print( '\nReturned ret < 0 -> creating new weight matrix and attempting again' )
                remove_weights( unts )
                continue
                
            ## If failed, check failure value for direction ##
            elif ret < ret_check:      # value is the minimum total epochs acceptable set at 10
                attempts += 1
                pstr = 'Failed Training With return value:' + str( ret ) + '\n'
                pstr += 'Epochs / Total Epochs:' + str( ret ) + ' / ' + str( num_epochs ) + '\n'
                pstr += 'Attempts completed: ' + str( attempts ) + ' / ' + str( max_attempts ) + '\n'
                ## Check to exit if Max attempts have been reached ##
                print( '\n' + pstr + '\n' )
                remove_weights( unts )
                continue
            
            attempts = 0    # Reset attempts -> was a succesful training session #
            count_completed += 1
            
            spc = ' '*4
            pstr = '\n' + '. '*50 + '\n'
            pstr += 'Finished ' + _exp_name + ' : '
            pstr += spc + 'Time: ' + str( datetime.now() ) + '\n'
            pstr += spc + 'Tests Completed: ' + str( count_completed ) + '\n'
            pstr += spc + 'Epochs Completed: ' + str( ret ) + '\n'
            print( pstr )
        
        print( ' '*25 + ' TRAINING COMPLETE \n' + '. '*50 )
        
    else:
        
        ret = _experiment_func(
            _input_size = input_size,
            _batch_size = batch_size,
            _epoch_size = epoch_size,
            _num_epochs = num_epochs,
            **spec
        )
        
''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    ## Versions ##
#    print('Tensorflow Version:',tf.__version__)
    _expdict = {}
    _expdict['mkygls'] = ( mkygls_experiment , 'Mackey Glass' )
    _expdict['cpymem'] = ( cpymem_experiment , 'Copy Memory' )
    
    args = sys.argv
    arglen = len( args )
    
    if arglen > 2:
        
        ## Run experiments ##
        now = datetime.now()
        print('Start Date & Time:\n' , now , '\n' )
        
        # Collect test variables #
        _name = args[1] if arglen > 1 else 'unknown'
        _active = args[2] if arglen > 2 else 'n'
        _units = getint( args[3] ) if arglen > 3 else 16
        _size = getint( args[4] ) if arglen > 4 else 50
        _batches = getint( args[5] ) if arglen > 5 else 25
        _epochs = getint( args[6] ) if arglen > 6 else 2
        _tstsz = getint( args[7] ) if arglen > 7 else 1
        _tstnme = args[8] if arglen > 8 else 'mkygls'
        
        if ( _units is None ) or ( _size is None ): print( 'ERROR: input units or size parameters are not integers.' )
        
        _epoch_size = _batches if _tstnme == 'cpymem' else None
        
        spec = build_spec( _units , _name , _active , 'O' , 'GU' , _size )
        run_experiment_logic(
            _expdict[_tstnme][0],
            _exp_name = _expdict[_tstnme][1],
            test_size = _tstsz,
            input_size = _size,
            batch_size = _batches,
            epoch_size = _epoch_size,
            num_epochs = _epochs,
            **spec
        )
        
        spc = ' '*4 + '- '
        
        sepstr = 'x '*60 + '\n'
        endstr = sepstr + '\n' + '  '*25 + ' TEST ENDED ' + '  '*25 + '\n'
        print( '\n'*3 + endstr*3 + '\n'*3 )
        
    elif arglen == 2:
        print( 'Removing weights sizes u' + args[1] )
        remove_weights( int ( args[1] ) )
    else:
        
        spc = 5
        
        _tstnme = 'mkygls'
        name_list = [ 'dft_test' ]
        act_list = [ 'm2' ]
        unit_sz_lst = [ 24 ]
        inpt_sz_lst = [ 32 ]
        
        for nme in name_list:
            for act in act_list:
                for unt in unit_sz_lst:
                    for isz in inpt_sz_lst:
                        spec = build_spec( unt , nme , act , 'O' , 'GU' , isz )
                        print( '\n'*spc + '-- * '*20 + '\n'*spc )
                        run_experiment_logic(
                            _expdict[_tstnme][0],
                            _exp_name = _expdict[_tstnme][1],
                            test_size = 1,
                            input_size = isz,
                            batch_size = 25,
                            epoch_size = None,
                            num_epochs = 100,
                            _testing = False,
                            **spec
                        )
                        print( '\n'*spc + '** - '*20 + '\n'*spc )
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    