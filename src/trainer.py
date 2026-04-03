
# Standard Imports
import os
import sys
import time
import math
import csv
from datetime import datetime
from inspect import isfunction

import json

# Library Imports
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import tensorflow as tf
import keras

## Local Imports ##
import proj_utils as utils
from proj_utils import _print , save_meta, load_meta


def build_weight_dict(model,name_list=[]):
    out_dict = {}
    for wgt in model.get_layer(index=1).weights:
        w = wgt.read_value().numpy()
        for nme in name_list:
            if nme in wgt.name:
                out_dict[nme] = w.copy()
    return out_dict


class ModelTrainer():
    
    ''' ModelTrainer '''
    def __init__(
                    self,
                    data_callback,
                    **kwargs
                 ):
        
        self.name = 'unknown_trainer'
        if 'name' in kwargs: self.name = kwargs['name']
        
        # Build directories
        self.trainer_dir = os.getcwd() + '\\training_runs\\' + self.name
#        self.chkpnt_dir = os.path.dirname(model_dir + '/results/checkpoints/training_1/cp.ckpt')
        
        self.main_metrics_dir = os.path.join( self.trainer_dir , 'master_metrics' )
        if not os.path.exists( self.main_metrics_dir ): os.makedirs( self.main_metrics_dir )
        
        # Setup training specific parameters
        self.batch_size = 1
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        assert self.batch_size > 0, self.name + ': batch_size cannot be less than 1.    batch_size = ' + str(self.batch_size)
        
        self.epoch_size = 1
        if 'epoch_size' in kwargs:
            self.epoch_size = kwargs['epoch_size']
        
        self.num_epochs = 1
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        assert self.num_epochs > 0, self.name + ': num_epochs cannot be less than 1.    num_epochs = ' + str(self.num_epochs)
        
        self.use_dataset = False
        if 'use_dataset' in kwargs: self.use_dataset = kwargs['use_dataset']
        
        # The training and testing data (can be either a list-of-list, np.ndarray, or tf.dataset)
        self.dataobj = data_callback( **kwargs )
        data = self.dataobj.generate()
        
        # Function callback for getting ground truth data matching prediction data
        self.groundtruth_callback = self.dataobj.groundtruth_generator
        
        # Each of training and testing are tuples of images and labels
        self.train_data, self.test_data = data.get_data()
        self.train_dataset, self.valid_dataset , self.test_dataset = None , None , None
        if self.use_dataset:
            self.train_dataset, self.valid_dataset , self.test_dataset = data.get_dataset()
        
        # Current model generated in train() from model_callbacks
        self.models = {}
        self.model_idx = 0


    ''' ModelTrainer '''
    def execute(
                 self,
                 model_callbacks,
                 model_directory='',
                 fit_valid_split = 0.0,
                 do_prediction=False,
                 do_evaluation=False,
                 verbose=1,
                 testing=None,
                 load_chkpnt = None,
                 safe_mode = True,
                 **model_spec
                ):
        
        if self.num_epochs > 0:
            retval = self.num_epochs
        else:
            return -1
        
        # Check input callbacks list
        if isfunction(model_callbacks): model_callbacks = [model_callbacks]
        if not isinstance(model_callbacks,list): model_callbacks = list(model_callbacks)
        
        # Train and Test Data
        trn_imgs, trn_lbls = self.train_data
        tst_imgs, tst_lbls = self.test_data
        
        if verbose > 0:
            print('\nTrainer.execute | Data Size:')
            print('Train Images:',trn_imgs.shape)
            print('Train Labels:',trn_lbls.shape)
            print(' Test Images:',tst_imgs.shape)
            print(' Test Labels:',tst_lbls.shape)
            print()
#        exit(0)
        
        tf.keras.backend.clear_session()
        
        # Run quick tests without training.
        if (testing is not None) and isinstance(testing, int) and (testing > 0):
            
            _pstr = '\n'*2 + '-'*20 + ' TESTING (' + str(testing) + ') ' + '-'*20 + '\n'*2
            print( _pstr*3 )
            
            assert len(model_callbacks) > 0, 'trying to test with trainer.py but the model callbacks list was empty'
            
            for model_callback in model_callbacks:
                
                # Clear the backend stuff.
                tf.keras.backend.clear_session()
                
                if len( model_spec ) == 0: model_spec = {}
                
                ''' CREATE MODEL '''
                model_struct = model_callback( trn_imgs[0].shape , trn_lbls[0].shape , self.batch_size , self.trainer_dir , **model_spec )
                
                assert 'model' in model_struct, 'trying to test with trainer.py but the model was not in model_struct after creation'
                
                # Generate sudo batch stuff
                
                bs = trn_imgs.shape[0] // self.batch_size
                sudo_batch = trn_imgs[0:bs*self.batch_size]
                rshparr = [bs,self.batch_size,sudo_batch.shape[-2],sudo_batch.shape[-1]]
                sudo_batch = np.reshape( sudo_batch , rshparr )
                
                for i in range(testing):
                    model_struct['model'](sudo_batch[i])
                    
                return -1
            
        else:
            
            rt_dict = utils.runtime_dict()
            
            # Itterate the model_callbacks list.
            #   Test for compatibility with the dataset.
            for model_callback in model_callbacks:
                
                # Clear the backend stuff.
                tf.keras.backend.clear_session()
                
                if len( model_spec ) == 0: model_spec = {}
                
                ''' CREATE MODEL '''
                model_struct = model_callback(
                    trn_imgs[0].shape,
                    trn_lbls[0].shape,
                    self.batch_size,
                    self.trainer_dir,
                    **model_spec
                )
                
                # Set initial model runtime
                model_struct['runtime'] = rt_dict
                
                ''' TEST DATASET AND MODEL COMPATIBILITY '''
                # Test for output shape compatibility
                #utils.ensure_output_shape(model_struct['model'], self.train_data)
                
                # Load dictionary item to trainer models dictionary.
                modstr = model_struct['name']+'_'+str(self.model_idx)
                self.models[modstr] = model_struct
                self.model_idx += 1
                
                
                model = model_struct['model']
                
                iniwgt_list, finwgt_list = None, None
                
                # Add a copy of the specified weights to the structure BEFORE training,
                #   if a list of names was provided in the model_callback functions 
                #   (see the models.py file for more details on where this is added)
                if 'wgtlst' in model_struct:
                    model_struct['iniwgt'] = utils.build_weight_dict( model , model_struct['wgtlst'] )
                
                if isinstance( load_chkpnt , str ):
                    model.load_weights( tf.train.latest_checkpoint( load_chkpnt ) )
                    
                model_struct['train_time_start'] = time.time()
                
                training_failure = False    # Set to True if training fails #
                
                if safe_mode:
                    
                    try:
                    
                        if self.use_dataset and ( self.train_dataset is not None ):
                            model_struct['hist'] = model.fit( 
                                self.train_dataset,
                                epochs = self.num_epochs,
                                steps_per_epoch = self.epoch_size,
                                validation_data = self.valid_dataset,
                                verbose = verbose,
                                shuffle = model_struct['shuffle'],
                                callbacks = model_struct[ 'callbacks' ]
                            )
                        else:
                            model_struct['hist'] = model.fit( 
                                x = trn_imgs, 
                                y = trn_lbls,
                                batch_size = self.batch_size,
                                epochs = self.num_epochs,
                                steps_per_epoch = self.epoch_size,
                                validation_split = fit_valid_split,
                                verbose = verbose,
                                shuffle = model_struct[ 'shuffle' ],
                                callbacks = model_struct[ 'callbacks' ]
                            )
                            
                    except:
                        
                        training_failure = True
                        print( '\n'*2 + 'Failed Training ' + model_struct['name'] + '\n'*2 )
                        
                        ## Reset to last successful Epoch ##
                        tf.keras.backend.clear_session()
                        if os.path.exists( model_struct['chkpnt_dir'] ):
                            model.load_weights( tf.train.latest_checkpoint( model_struct['chkpnt_dir'] ) )
                            
                else:
                    
                    if self.use_dataset and ( self.train_dataset is not None ):
                        model_struct['hist'] = model.fit( 
                            self.train_dataset,
                            epochs = self.num_epochs,
                            validation_data = self.valid_dataset,
                            verbose = verbose,
                            shuffle = model_struct['shuffle'],
                            steps_per_epoch = self.epoch_size,
                            callbacks = model_struct[ 'callbacks' ]
                        )
                    else:
                        model_struct['hist'] = model.fit( 
                            x = trn_imgs, 
                            y = trn_lbls,
                            batch_size = self.batch_size,
                            epochs = self.num_epochs,
                            validation_split = fit_valid_split,
                            verbose = verbose,
                            shuffle = model_struct['shuffle'],
                            steps_per_epoch = self.epoch_size,
                            callbacks = model_struct[ 'callbacks' ]
                        )
                        
                ## Reset to last successful Epoch ##
                tf.keras.backend.clear_session()
                if os.path.exists( model_struct['chkpnt_dir'] ):
                    lstchkpt = tf.train.latest_checkpoint( model_struct['chkpnt_dir'] )
                    model.load_weights( lstchkpt )
                    model_struct['best_chkpnt'] = lstchkpt.split('\\')[-1][3:7]
                    print( 'Loaded Best Checkpoint:' , model_struct['best_chkpnt'] )
                
                ## Collect Batch, Epoch Information ##
                model_struct['batch_size'] = int( self.batch_size )
                model_struct['epoch_size'] = int( self.num_epochs )
                if 'hist' in model_struct:
                    model_struct['epoch_count'] = model_struct['hist'].params['epochs']
                    model_struct['steps_count'] = model_struct['hist'].params['steps']
                else:
                    model_struct['epoch_count'] = 0
                    model_struct['steps_count'] = 0
                
                ## Print Training History Information to Command Window ##
                if verbose > 0 and 'hist' in model_struct:
                    print('Train History:')
                    for k , v in model_struct['hist'].history.items():
                        vals = np.asarray( v )
                        mx = np.round( np.max( vals ) , 4 )
                        mn = np.round( np.min( vals ) , 4 )
                        av = np.round( np.mean( vals ) , 4 )
                        print( '\t'+k+':\n\t','max:', mx,' min:',mn,' mean:',av,'\n' )
                        
                        ## Test for nan values and exit early if there are too many else make them zero ##
                        nanvals = np.count_nonzero( np.isnan( vals ).astype( int ) )
                        if nanvals > ( vals.shape[0] / 2. ):
                            print( k.lower().capitalize() + ' had too many \'nan\' values, exiting early' )
                            return -1
                            
                if training_failure: retval =  model_struct['epoch_count']
                    
                model_struct['train_time_end'] = time.time()
                
                # Add a copy of the specified weights to the structure AFTER training.
                #   (This is the same as above, but after the weights have been updated)
                if 'wgtlst' in model_struct:
                    model_struct['finwgt'] = utils.build_weight_dict( model , model_struct['wgtlst'] )
                    
                # Make sure requested batch-shape is compatible to reshape before prediction
                assert (tst_imgs.shape[0] % self.batch_size) == 0, 'total image dataset size must be divisible by the set batch-size.'
                
                if do_prediction:
                    
                    model_struct['pred_time_start'] = time.time()
                    predsteps = model_struct['pred_steps'] if 'pred_steps' in model_struct else 1e3
                    
                    ''' Predict on output '''
                    if verbose > 0: print('\nPredict:')
                    if safe_mode:
                    
                        
                        try:
                            if self.use_dataset and ( self.test_dataset is not None ):
                                model_struct['pred'] = model.predict(
                                    self.test_dataset,
                                    verbose = verbose
                                )
                            else:
                                model_struct['pred'] = model.predict(
                                    tst_imgs,
                                    batch_size = self.batch_size,
                                    #steps=predsteps,
                                    verbose = verbose
                                )
                        except:
                            print( '\n'*2 + 'Failed Prediction ' + model_struct['name'] + '\n'*2 )
                            if retval > 1: retval -= 1
                    else:
                        
                        if self.use_dataset and ( self.test_dataset is not None ):
                            model_struct['pred'] = model.predict(
                                self.test_dataset,
                                verbose = verbose
                            )
                        else:
                            model_struct['pred'] = model.predict(
                                tst_imgs,
                                batch_size = self.batch_size,
                                #steps=predsteps,
                                verbose = verbose
                            )
                    
                    pred = np.asarray( model_struct['pred'] ) if not isinstance( model_struct['pred'] , np.ndarray ) else model_struct['pred']
                    ## If prediction fails , return failure and create a new matrix ##
                    if np.mean( pred ) == 0.0:
                        print( 'Prediction failed ... returning to create new weight matrix' )
                        return -1
                    
                    if verbose > 0: print('\n')
                    if verbose > 1 and 'pred' in model_struct: print('pred:',model_struct['pred'])
                    
                    model_struct['pred_time_end'] = time.time()
                    
                    
                if do_evaluation:
                    
                    start_time = time.time()
                    
                    model_struct['eval_time_start'] = time.time()
                    evalsteps = model_struct['eval_steps'] if 'eval_steps' in model_struct else 1e3
                    
                    if verbose > 0: print('\nEvaluate:')
                    if safe_mode:
                        
                        try:
                            if self.use_dataset and ( self.test_dataset is not None ):
                                model_struct['eval'] = model.evaluate(
                                    self.test_dataset,
                                    verbose = verbose
                                )
                            else:
                                model_struct['eval'] = model.evaluate(
                                    tst_imgs,
                                    tst_lbls,
                                    verbose = verbose,
                                    batch_size = self.batch_size,
                                    #steps=evalsteps
                                )
                        except:
                            print( '\n'*2 + 'Failed Evaluation ' + model_struct['name'] + '\n'*2 )
                            if retval > 1: retval -= 1
                            
                    else:
                        
                        if self.use_dataset and ( self.test_dataset is not None ):
                            model_struct['eval'] = model.evaluate(
                                self.test_dataset,
                                verbose = verbose
                            )
                        else:
                            model_struct['eval'] = model.evaluate(
                                tst_imgs,
                                tst_lbls,
                                verbose = verbose,
                                batch_size = self.batch_size,
                                #steps=evalsteps
                            )
                        
                    if verbose > 0: print('\n')
                    if verbose > 1 and 'eval' in model_struct: print( 'eval:' , model_struct['eval'] )
                    
                    model_struct['eval_time_end'] = time.time()
                    
        return retval


    ''' ModelTrainer 
        Saves each model ran through the trainer.
        In the specified model directory.'''
    def save(
              self,
              max_plot_len = 1024,
              show_plot = False,
              save_plot = False,
              show_weights = False,
              save_weights = False,
              verbose = 0,
              datnme = None
             ):
        
        if len( self.models ) == 0:
            return self
        
        csvhdr = []     # Header of the CSV
        csvstr = []     # Normal CSV stuff
        grpidx = 2      # Index to group the Average and Evaluation elements
        
        def setcsv( msg , val , pos = -1 ):
            if pos < 0:
                csvhdr.append( msg )
                csvstr.append( str( val ) )
            else:
                csvhdr.insert( pos , msg )
                csvstr.insert( pos , str( val ) )
            
        for k, v in self.models.items():
            
            tf.keras.backend.clear_session()
            
            print('Saving Model:')
            print('  name:', v['name'] )
            
            if 'name' in v: setcsv( 'name' , v['name'] )
            
            # For saving a text summary file.
            readme = {}
            
            readme['dataset'] = self.name.capitalize()
            data_attr = self.dataobj.get_attributes()
            for dk, dv in data_attr.items():
                readme[dk] = dv
            
            # Get generation
            gen = int(v['gdir'].split('_')[-1])
            v['gen'] = gen
            print('  Generation:', gen )
            setcsv( 'Generation' , v['gen'] )
            
            if 'params' in v:
                for prmnme , prmval in v['params'].items():
                    prmstr = prmnme.lower().split('_')
                    prmstr = ' '.join( [ ps.capitalize() for ps in prmstr ] )
                    setcsv( prmstr , prmval )
            
            if 'batch_size' in v: setcsv( 'Batch Size' , v['batch_size'] )
            if 'epoch_size' in v: setcsv( 'Epoch Size' , v['epoch_size'] )
            if 'steps_count' in v: setcsv( 'Epoch Steps' , v['steps_count'] )
            if 'epoch_count' in v: setcsv( 'Epochs Trained' , v['epoch_count'] )
            
            ''' - - - - - SAVE MODEL - - - - - '''
            
            # Save the model in the gen_## file directory
            mfiledir = os.path.join(v['gdir'],'model')
            if not os.path.exists(mfiledir):
                os.makedirs( mfiledir )
            
            ''' - - - - - SAVE TRAINING RESULTS - - - - - '''
            # Save the model results in the gen_## file directory
            rfiledir = os.path.join(v['gdir'],'results')
            if not os.path.exists(rfiledir):
                os.makedirs( rfiledir )
            
            # Get Model Summary as string.
            sumstr = []
            v['model'].summary( print_fn = lambda x: sumstr.append(x) , expand_nested=True , show_trainable=True )
            readme['summary'] = '\n'.join( sumstr )
            
            ## Create and or Setup main metrics directory ##
            main_metrics_filename = v['name'] + '_gen_' + str(gen) + '_'
            
            ''' - - - - - SAVE MODEL WEIGHTS AND COMPARISON - - - - - '''
            
            ## Plot and Save Initial Weight Matrices ##
            if 'iniwgt' in v:
                
                for wk , wv in v['iniwgt'].items():
                    
                    ## Get weight Filename & Create Path ##
                    wfn = 'iniwgt_' + wk
                    
                    wgtdct = {}
                    
                    ## Build save dictionary, Convert if complex valued to save Real & Imag in separate files ##
                    if np.iscomplexobj( wv ):
                        wgtdct['real'] = wv.real.copy()
                        wgtdct['imag'] = wv.imag.copy()
                    else:
                        wgtdct['float'] = wv.copy()
                        
                    ## Plot & Save Weights ##
                    for wvn , wvm in wgtdct.items():
                        utils.plot(
                            wvm,
                            title = 'Initial Weight ' + ' ' + wk.capitalize() + ' (' + wvn + ') ' + '  gen: ' + str( gen ),
                            xlabel = 'X',
                            ylabel = 'Y',
                            show = False,
                            save = save_weights,
                            dir = rfiledir,
                            name = wfn + '_' + wvn
                        )
                        
#                    _fn = wfn + '_' + wvn
                    filename = os.path.join( mfiledir , wfn+'.npy' )
                    np.save( filename , wv , allow_pickle = True )
                    v[ wfn + '_dir' ] = filename
                    
                    #'''
                    if len( list( wv.shape ) ) == 2 and wv.shape[0] == wv.shape[1]:

                        ## Save Initial Eigen Values for Weights ##
                        evls , evcs = np.linalg.eig( wv )
                        
                        ## Get eigen vector weight Filename & Create Path ##
                        efn = 'inievcs_' + wk
                        
                        ## Build save dictionary, Convert if complex valued to save Real & Imag in separate files ##
                        wgtdct = {}
                        if np.iscomplexobj( evcs ):
                            wgtdct['real'] = evcs.real.copy()
                            wgtdct['imag'] = evcs.imag.copy()
                        else:
                            wgtdct['float'] = evcs.copy()
                            
                        ## Plot & Save Weights ##
                        for wvn , wvm in wgtdct.items():
                            utils.plot(
                                wvm,
                                title = 'Initial Eigen Vectors ' + ' ' + wk.capitalize() + ' (' + wvn + ') ' + '  gen: ' + str( gen ),
                                xlabel = 'X',
                                ylabel = 'Y',
                                show = False,
                                save = save_weights,
                                dir = rfiledir,
                                name = efn + '_' + wvn
                            )
                            ## Save Weights ##
                            if save_weights:
                                _fn = efn + '_' + wvn
                                filename = os.path.join( mfiledir , _fn+'.npy' )
                                np.save( filename , wvm , allow_pickle = True )
                                v[ _fn + '_dir' ] = filename
                        
                        ## Get eigen vector weight Filename & Create Path ##
                        efn = 'inievls_' + wk
                        
                        ## Build save dictionary, Convert if complex valued to save Real & Imag in separate files ##
                        wgtdct = {}
                        if np.iscomplexobj( evls ):
                            wgtdct['real'] = evls.real.copy()
                            wgtdct['imag'] = evls.imag.copy()
                        else:
                            wgtdct['float'] = evls.copy()
                            
                        ## Plot & Save Weights ##
                        for wvn , wvm in wgtdct.items():
                            utils.plot(
                                wvm,
                                title = 'Initial Eigen Values ' + ' ' + wk.capitalize() + ' (' + wvn + ') ' + '  gen: ' + str( gen ),
                                xlabel = 'X',
                                ylabel = 'Y',
                                show = False,
                                save = save_weights,
                                dir = rfiledir,
                                name = efn + '_' + wvn
                            )
                                
                        ## Save Weights ##
                        if save_weights:
                            filename = os.path.join( mfiledir , efn+'.npy' )
                            np.save( filename , wvm , allow_pickle = True )
                            v[ efn + '_dir' ] = filename
                        
                    # < if np.rank( wv ) == 2 and wv.shape[0] == wv.shape[1]: >
                    #'''
                    
            ## Plot and Save Final Weight Matrices ##
            if 'finwgt' in v:
                
                for wk , wv in v['finwgt'].items():
                    
                    ## Get weight Filename & Create Path ##
                    fn = 'finwgt_' + wk
                    
                    ## Build save dictionary, Convert if complex valued to save Real & Imag in separate files ##
                    if np.iscomplexobj( wv ):
                        wgtdct['real'] = wv.real.copy()
                        wgtdct['imag'] = wv.imag.copy()
                    else:
                        wgtdct['float'] = wv.copy()
                        
                    ## Plot & Save Weights ##
                    for wvn , wvm in wgtdct.items():
                        utils.plot(
                            wvm,
                            title = 'Final Weight ' + ' ' + wk.capitalize() + ' (' + wvn + ') ' + '  gen: ' + str( gen ),
                            xlabel = 'X',
                            ylabel = 'Y',
                            show = False,
                            save = save_weights,
                            dir = rfiledir,
                            name = fn + '_' + wvn
                        )
                        
                    ## Always Save Weights ##
                    filename = os.path.join( mfiledir , fn+'.npy' )
                    np.save( filename , wvm , allow_pickle = True )
                    v[ fn + '_dir' ] = filename
                    
                    #'''
                    if len( list( wv.shape ) ) == 2 and wv.shape[0] == wv.shape[1]:
                    
                        ## Save Final Eigen Values for Weights ##
                        evls , evcs = np.linalg.eig( wv )
                        
                        ## Get eigen vector weight Filename & Create Path ##
                        efn = 'finevcs_' + wk
                        
                        ## Build save dictionary, Convert if complex valued to save Real & Imag in separate files ##
                        wgtdct = {}
                        if np.iscomplexobj( evcs ):
                            wgtdct['real'] = evcs.real.copy()
                            wgtdct['imag'] = evcs.imag.copy()
                        else:
                            wgtdct['float'] = evcs.copy()
                            
                        ## Plot & Save Weights ##
                        for wvn , wvm in wgtdct.items():
                            utils.plot(
                                wvm,
                                title = 'Final Eigen Vectors ' + ' ' + wk.capitalize() + ' (' + wvn + ') ' + '  gen: ' + str( gen ),
                                xlabel = 'X',
                                ylabel = 'Y',
                                show = False,
                                save = save_weights,
                                dir = rfiledir,
                                name = efn + '_' + wvn
                            )
                                
                        ## Save Weights ##
                        if save_weights:
                            filename = os.path.join( mfiledir , efn+'.npy' )
                            np.save( filename , wvm , allow_pickle = True )
                            v[ efn + '_dir' ] = filename
                        
                        ## Get eigen vector weight Filename & Create Path ##
                        efn = 'finevls_' + wk
                        
                        ## Build save dictionary, Convert if complex valued to save Real & Imag in separate files ##
                        wgtdct = {}
                        if np.iscomplexobj( evls ):
                            wgtdct['real'] = evls.real.copy()
                            wgtdct['imag'] = evls.imag.copy()
                        else:
                            wgtdct['float'] = evls.copy()
                            
                        ## Plot & Save Weights ##
                        for wvn , wvm in wgtdct.items():
                            utils.plot(
                                wvm,
                                title = 'Final Eigen Values ' + ' ' + wk.capitalize() + ' (' + wvn + ') ' + '  gen: ' + str( gen ),
                                xlabel = 'X',
                                ylabel = 'Y',
                                show = False,
                                save = save_weights,
                                dir = rfiledir,
                                name = efn + '_' + wvn
                            )
                                
                        ## Save Weights ##
                        if save_weights:
                            filename = os.path.join( mfiledir , efn+'.npy' )
                            np.save( filename , wvm , allow_pickle = True )
                            v[ efn + '_dir' ] = filename
                            
                    # < if np.rank( wv ) == 2 and wv.shape[0] == wv.shape[1]: > 
                    #'''
                    
            ## Calculate & Plot Matrix Changes ##
            if ( 'iniwgt' in v ) and ( 'finwgt' in v ) and save_weights:
                
                for wgtnme , iwgt in v['iniwgt'].items():
                    
                    ## Find final weight ## 
                    if wgtnme not in v['finwgt']: continue
                    
                    ## Collect finished weight ##
                    fwgt = v['finwgt'][wgtnme]
                    
                    ## Set file name header ##
                    wfn = 'weight_' + wgtnme
                    
                    ## Calculate Differences ##
                    difdct = {}
                    difdct['Standard Difference'] = ( iwgt - fwgt ).copy()
                    difdct['Absolute Difference'] = np.abs( iwgt - fwgt ).copy()
                    
                    lenchk = ( len( iwgt.shape ) > 1 ) and ( len( fwgt.shape ) > 1 )
                    if lenchk and fwgt.shape[-2] == iwgt.shape[-1]:
                        difdct['Forward Commutator'] = ( fwgt @ iwgt - iwgt @ fwgt ).copy()
                        difdct['Backward Commutator'] = ( iwgt @ fwgt - fwgt @ iwgt ).copy()
                    
                    for wttl , wdif in difdct.items():
                        
                        dfn = wfn + '_' + wttl.lower().replace( ' ' , '_' )
                        
                        ## Build save dictionary, Convert if complex valued to save Real & Imag in separate files ##
                        if np.iscomplexobj( wdif ):
                            wgtdct = { 'real' : wdif.real.copy() , 'imag' : wdif.imag.copy() }
                        else:
                            wgtdct = { 'float' : wdif.copy() }
                        
                        for wvn , wvm in wgtdct.items():
                            pfn = dfn + '_' + wvn
                            utils.plot(
                                wvm,
                                title = wttl + ' - ' + wk.capitalize() + ' (' + wvn + ') ' + ' - Gen: ' + str(gen),
                                xlabel = 'X',
                                ylabel = 'Y',
                                show = False,
                                save = save_weights,
                                dir = rfiledir,
                                name = pfn
                            )
                            
                        ## Save Weights ##
                        if save_weights:
                            filename = os.path.join( mfiledir , pfn + '.npy' )
                            np.save( filename , wvm , allow_pickle = True )
                            v[ pfn + '_dir' ] = filename
                            
                            
            def get_time_str( msg ):
                rt = v[ msg + '_end' ] - v[ msg + '_start' ]
                rt_hrs = rt // 3600
                v[ msg + '_hrs'] = rt_hrs
                rt_min = ( rt - 3600 * rt_hrs ) // 60
                v[ msg + '_min'] = rt_min
                rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
                rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
                v[ msg + '_rem'] = rt_rem
                rt_sec = math.floor( rt_sec )
                v[ msg + '_sec'] = rt_sec
                return '{:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)'.format( rt_hrs , rt_min , rt_sec , rt_rem )
            
            ## Print run-times
            if 'train_time_start' in v and 'train_time_end' in v: readme['Training RunTime: '] = get_time_str( 'train_time' )
            if 'pred_time_start' in v and 'pred_time_end' in v: readme['Prediction RunTime: '] = get_time_str( 'pred_time' )
            if 'eval_time_start' in v and 'eval_time_end' in v: readme['Evalutaion RunTime: '] = get_time_str( 'eval_time' )
            
            readme[''] = ''
            readme['Metrics'] = ''
            
            readme_metrics = []
            if 'hist' in v:
                
                hist = v['hist']
                v['fit'] = hist.params
                metrics = []
                legend = []
                for hk, hv in hist.history.items():
                    v[hk] = hv
                    nphv = np.asarray( hv )
                    nphv = np.nan_to_num( nphv )
                    filename = os.path.join(rfiledir,'train_'+hk+'.npy')
                    np.save( filename , nphv , allow_pickle=True )
                    legend.append( hk )
                    
                    np.save(
                        os.path.join( self.main_metrics_dir , main_metrics_filename + 'train_'+hk+'.npy' ),
                        nphv,
                        allow_pickle=True
                    )
                    
#                    if 'time' not in hk: metrics.append( nphv )
                    if 'val_' not in hk: readme_metrics.append( hk )      # Add string for later text file 'readme'
                    
                    if 'time' in hk:
                        setcsv( 'Ave Train Time (excel)' , np.mean( nphv ) / 84600 , grpidx )
                        grpidx+=1
                        continue
                    
                    # Add to metrics all but the Time values ##
                    metrics.append( nphv )
                    
                    tmpstr = 'Training '
                    mx = np.round( np.max( nphv ) , 5 )
                    mxst = tmpstr + 'Max: ' + hk
                    readme[ mxst ] = mx
                    #setcsv( mxst , mx )
                    
                    mn = np.round( np.min( nphv ) , 5 )
                    mnst = tmpstr + 'Min: ' + hk
                    readme[ mnst] = mn
                    #setcsv( mnst , mn )
                    
                    av = np.round( np.mean( nphv ) , 5 )
                    avst = tmpstr + 'Ave: ' + hk
                    readme[ avst ] = av
                    if 'val_' in hk:
                        setcsv( avst , av , pos = grpidx )
                        grpidx += 1
                    else:
                        setcsv( avst , av )
                        
                    
                        
                        
                x_vals = np.arange( len( metrics[0] ) )
                y_vals = metrics
                
                # Plot Metrics
                utils.plot( 
                    x_vals,
                    y_vals,
                    colors=['b','g','r','m'],
                    title='Metric vs. Epoch - '+v['name'] + ' - Gen: ' + str(gen),
                    xlabel='Epochs',
                    ylabel='Metrics',
                    legend=legend,
                    show=show_plot,
                    save=save_plot,
                    dir=rfiledir,
                    name='metrics'
                )
            
            ''' - - - - - SAVE EVALUATION RESULTS - - - - - '''
            # Prediction values and plot
            if 'pred' in v:
                
                if 'mkygls' in datnme :
                    
                    try:
                        # Save predictions as np.ndarray in file
                        filename = os.path.join(rfiledir,'pred.npy')
                        np.save( filename , np.asarray( v['pred'] ) , allow_pickle = True )
                        v['pred_dir'] = filename
                            
                        np.save(
                            os.path.join( self.main_metrics_dir , main_metrics_filename + 'pred.npy' ),
                            np.asarray( v['pred'] ),
                            allow_pickle=True
                        )
                        
                        # Plot and save results based on ground truth
                        gt , est = self.groundtruth_callback( v['pred'] , max_plot_len )
                        ave = np.mean( est ) * np.ones_like( est )
                        x_vals = np.arange( gt.shape[0] )
                        legend=['Ground Truth','Predicted']
                        utils.plot(
                                    x_vals,
                                    [ gt , est , ave ],
                                    colors=['b','r','g'],
                                    linestyles=['-','--','--'],
                                    title=self.name.capitalize() + ' Results - '+v['name'] + ' - Gen: ' + str(gen),
                                    xlabel='Time',
                                    legend=legend,
                                    show=show_plot,
                                    save=save_plot,
                                    dir=rfiledir,
                                    name=v['name']
                                   )
                    except:
                        pass
                
                elif 'cpymem' in datnme :
                    
                    # Save predictions as np.ndarray in file
                    filename = os.path.join(rfiledir,'pred.npy')
                    np.save( filename , np.asarray( v['pred'] ) , allow_pickle = True )
                    v['pred_dir'] = filename
                    
                    np.save(
                        os.path.join( self.main_metrics_dir , main_metrics_filename + 'pred.npy' ),
                        np.asarray( v['pred'] ),
                        allow_pickle=True
                    )
                    
                    # Plot and save results based on ground truth
                    gt , est = self.groundtruth_callback( v['pred'] , max_plot_len )
                    
                    seqlen = list( gt.shape )[1]
                    
                    # Gen 10 random values between 0 and pred size
                    idxs = np.random.randint( 0 , gt.shape[0] , size = 5 )
                    
                    _showplot = show_plot
                    
                    for id in idxs:
                        
                        estshp = list( est[id].shape )
                        gtshp = list( gt[id].shape )
                        
                        # Check if images need to be transposed for horizontal view #
                        estplt = est[id].T if estshp[0] > estshp[1] else est[id]
                        gtplt = gt[id] if gtshp[0] > gtshp[1] else gt[id]
                        
                        # Shorten images to fit in 100 timesteps if it's too long #
                        if estplt.shape[-1] > 100:
                            estplt = estplt[::,-100::]
                            gtplt = gtplt[::,-100::]
                            
                        fig , axs = plt.subplots( 2 , 1 )
                        fig.suptitle('Copy Memory Task (Max Sequence Length ' + str( seqlen ) + ') Index: ' + str( id ) )
                        fig.supxlabel('Time - - - - >')
                        
                        img0 = axs[0].imshow( estplt )
                        axs[0].set_ylabel('Predition')
                        axs[0].set_aspect('auto')
                        
                        axs[1].imshow( gtplt )
                        axs[1].set_ylabel('Ground Truth')
                        axs[1].set_aspect('auto')
                            
                        fig.colorbar( mappable = img0 , ax = axs , orientation='vertical' , fraction = 0.1 )
                        
                        if save_plot:
                            pathname = os.path.join( rfiledir , 'pred_plot_idx_' + str( id ) + '.png' )
                            plt.savefig( pathname , dpi = 300 )
                        
                        # Only show the plot once
                        if _showplot:
                            plt.show()
                            _showplot = False
                        
                        plt.close()
                        
                else:
                    print('\n[trainer|save|860] WARNING: unknown datnme for model ' + v['name'] + '\n')
            
            grpidx2 = grpidx
            
            # Save Evaluation
            if 'eval' in v:
                
                if not isinstance( v['eval'] , list ): v['eval'] = [ v['eval'] ]
                
                evlstrs = list(v['hist'].history.keys())[0:len(v['eval'])]
                
                for i in range( len( evlstrs ) ):
                    evlstr = 'Eval ' + evlstrs[i]
                    evlval = round( v['eval'][i] , 5 )
                    readme[evlstr] = evlval
                    setcsv( evlstr ,evlval , grpidx )
                    grpidx += 1
                
                eval = np.asarray( v['eval'] )
                np.save(
                        os.path.join( rfiledir ,'eval.npy'),
                        eval,
                        allow_pickle=True
                        )
                v['eval_dir'] = filename
                
                np.save(
                    os.path.join( self.main_metrics_dir , main_metrics_filename + 'pred.npy' ),
                    eval,
                    allow_pickle=True
                )
                
            # Write a summary that contains all relevant information to the generation folder
            rmmw = utils.readme_writer( filename = 'Summary' , filedir = v['gdir'] , **readme )
            if verbose > 1: print( rmmw )
            
            # Delete all non-json-able dictionary elements.
            if 'model' in v: del v['model']
            if 'callbacks' in v: del v['callbacks']
            if 'hist' in v: del v['hist']
            if 'pred' in v: del v['pred']
            if 'eval' in v: del v['eval']
            if 'finwgt' in v: del v['finwgt']
            if 'iniwgt' in v: del v['iniwgt']
            
#            if 'gdir' in v : setcsv( 'Origional Directory' , v['gdir'] )
            
            csvdir = os.path.join( v['dir'] , 'metrics.csv' )
            
            ## Check Load and Save the Model Meta File ##
            model_meta = load_meta( v['dir'] )
            if model_meta['csvset'] == 0:
                with open( csvdir , 'w' ) as csvfile:
                    csvfile.write( ','.join( csvhdr ) + ',\n' )
                    csvfile.write( ','.join( csvstr ) + ',\n' )
                model_meta['csvset'] = 1
            else:
                with open( csvdir , 'a' ) as csvfile:
                    csvfile.write( ','.join( csvstr ) + ',\n' )
            ret = save_meta( model_meta , v['dir'] )
            
            ''' Saving Training Meta '''
            # Save the remaining structure
            jsonstr = json.dumps( v )
            filename = os.path.join(v['gdir'],'meta.json')
            with open(filename,'w') as fileobj:
                fileobj.write(jsonstr)
                
        return self



