
# Standard Imports
import os
import sys
from datetime import datetime
import random

# Library Imports
import numpy as np

import matplotlib.pyplot as plt

# Special Imports #
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.datasets import mnist

import proj_utils as utils
from proj_utils import _print

'''
TODO:
- Make so that datasets can be made into the tf.keras.dataset.Dataset() type class.
    - Use a generator for this
    - Can load information with this
    - Requires changes to the trainer.
'''


class DataGenerator:
    
    ''' DataGenerator '''
    def __init__( self,
                  **kwargs
                 ):
        
        # Holds the data generated from the generate() function (i.e., not manipulated in to size,batch,epochs,...,etc.)
        self.raw_data = None
        self.raw_size = None
        self.raw_shape = ()
        
        # Holds the training and test data for model
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        
        # 2-dim tuple of sizes, matching the training and testing, images and labels.
        self.shape = None
        
        # Ground truth data for plotting and confirmation
        self.gndtru = None
        
        # Training & Testing specific values
        self.input_size = 0
        if 'input_size' in kwargs:
            self.input_size = kwargs['input_size']
            del kwargs['input_size']
        assert self.input_size > 0, '\ninput_size must be set and greater than 0'
            
        # Training & Testing specific values
        self.time_steps = self.input_size
        if 'time_steps' in kwargs:
            self.time_steps = kwargs['time_steps']
            del kwargs['time_steps']
        assert self.time_steps > 0, '\ntime_steps must be set and greater than 0'
        
        self.output_size = 0
        if 'output_size' in kwargs:
            self.output_size = kwargs['output_size']
            del kwargs['output_size']
        assert self.output_size > 0, '\noutput_size must be set and greater than 0'
        
        self.batch_size = 0
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
            del kwargs['batch_size']
        assert self.batch_size > 0, '\nbatch_size must be set and greater than 0'
        
        self.epoch_size = 1
        if 'epoch_size' in kwargs:
            self.epoch_size = kwargs['epoch_size']
            del kwargs['epoch_size']
        if self.epoch_size == 0 or self.epoch_size is None: self.epoch_size = 1
        
        self.num_epochs = 0
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
            del kwargs['num_epochs']
        assert self.num_epochs > 0, '\nnum_epochs must be set and greater than 0'
        
        self.test_split = 0.25
        if 'test_split' in kwargs:
            self.test_split = kwargs['test_split']
            del kwargs['test_split']
        assert self.test_split > 0.0, '\ntest_split must be set and greater than 0'
        assert self.test_split < 1.0, '\ntest_split must be set and less than 1.0'
        
        self.valid_split = 0.0
        if 'valid_split' in kwargs:
            self.valid_split = kwargs['valid_split']
            del kwargs['valid_split']
            assert self.valid_split >= 0.0, '\nvalid_split must be set and greater than or equal to 0'
            assert self.valid_split < 1.0, '\nvalid_split must be set and less than 1.0'
        
        self.use_dataset = False
        if 'use_dataset' in kwargs:
            self.use_dataset = kwargs['use_dataset']
            del kwargs['use_dataset']
        
        # Set the dtype if not defined in subclasses
        self.dtype = np.float32
        if 'dtype' in kwargs: self.dtype = kwargs['dtype']
        
        # Concatenates the name for subclasses
        self.name = 'DataGenerator'
        if 'name' in kwargs:
            self.name += '_' + kwargs['name']


    ''' DataGenerator '''
    def generate( self):
        # Placeholder for subclass
        return self


    ''' DataGenerator '''
    def groundtruth_generator( self , preds , maxlen=-1 ):
        
        preds = preds.copy()
        
        if not isinstance(preds,np.ndarray):
            preds = np.asarray(preds)
        
        gndtru = self.gndtru
        if not isinstance(gndtru,np.ndarray):
            gndtru = np.asarray(gndtru)
        
        stp = self.step_size
        preds = preds[::,-stp::]
        
        gndtru = gndtru.flatten()
        preds = preds.flatten()
        
        trulen = gndtru.shape[0]
        prdlen = preds.shape[0]

        minlen = min(trulen,prdlen)        
        
        # Ensure we have the same length
        if trulen != prdlen:
            gndtru = gndtru[0:minlen]
            preds = preds[0:minlen]
        
        # Limits the maximum output. (Used for plotting.)
        if maxlen > 0 and minlen > maxlen:
            gndtru = gndtru[0:maxlen]
            preds = preds[0:maxlen]
        
        return ( gndtru , preds )

    ''' DataGenerator '''
    def set_data(self, train_images, train_labels, test_images, test_labels):
        
        # Tests for input parameters, shapes, matching class parameters, ... etc.
        if  (train_images is None) or (train_labels is None) or (test_images is None) or (test_labels is None):
            return None
        
        # Ensure data-set outer dimension is divisible by the batch-size
        # Total-size of data-set    -> Check types
        if isinstance( train_images , np.ndarray ):
            # Check if image and label sets are divisible by the batch size; else, adjust them.
            train_remainder = train_images.shape[0] % self.batch_size
            test_remainder = test_images.shape[0] % self.batch_size
            if (train_remainder > 0) or (test_remainder > 0):
                new_train_size = train_images.shape[0] - train_remainder
                new_test_size = test_images.shape[0] - test_remainder
                train_images = train_images[0:new_train_size]
                train_labels = train_labels[0:new_train_size]
                test_images = test_images[0:new_test_size]
                test_labels = test_labels[0:new_test_size]
        
        # Set the final size of the datasets
        train_shape = ( train_images.shape , train_labels.shape )
        test_shape = ( test_images.shape , test_labels.shape )
        self.shape = ( train_shape , test_shape )
        
        self.train_images , self.train_labels = train_images , train_labels
        self.test_images , self.test_labels = test_images , test_labels
        
        return self

    def get_dataset( self ):
        if ( self.train_dataset is not None ) and ( self.test_dataset is not None ) and ( self.valid_dataset is not None ):
            return self.train_dataset , self.valid_dataset , self.test_dataset

    ''' DataGenerator '''
    def get_data(self):
        if (self.train_images is None) or (self.train_labels is None) or (self.test_images is None) or (self.test_labels is None):
            return ((None,None),(None,None))
        train = (self.train_images, self.train_labels)
        test = (self.test_images, self.test_labels)
        return (train, test)
    
    
    ''' DataGenerator '''
    def get_shape(self):
        return self.shape

    ''' DataGenerator '''
    def get_attributes(self,attr={}):
        attr['input_size'] = self.input_size
        attr['output_size'] = self.output_size
        attr['raw_size'] = self.raw_size
        attr['batch_size'] = self.batch_size
        attr['epoch_size'] = self.epoch_size
        attr['num_epochs'] = self.num_epochs
        attr['test_split'] = str(self.test_split*100)+'%'
        attr['use_dataset'] = self.use_dataset
        attr['dtype'] = self.dtype
        (trnimg,trnlbl),(tstimg,tstlbl)=self.shape
        attr['shape'] = '\n  Train Images: '+str(trnimg)+'\n  Train Labels: '+str(trnlbl)
        attr['shape'] += '\n  Test Images: '+str(tstimg)+'\n  Test Labels: '+str(tstlbl)
        return attr






class MackeyGlassGenerator( DataGenerator ):


    ''' MackeyGlass Generator '''
    def __init__( self, **kwargs):
        
        super().__init__( **kwargs )
        
        self.raw_size = 4096
        self.raw_data = None
        
        self.tao = 30
        if 'tao' in kwargs:
            self.tao = kwargs['tao']
            del kwargs['tao']
        
        self.delta_x = 10
        if 'delta_x' in kwargs:
            self.delta_x = kwargs['delta_x']
            del kwargs['delta_x']
        
        self.step_size = 1
        if 'step_size' in kwargs:
            self.step_size = kwargs['step_size']
            del kwargs['step_size']
        
    ''' MackeyGlass Generator '''
    def generate( self ):
        
        # TODO: Transfer these to the lower code section.
        input_size = self.input_size
        output_size = self.output_size
        batch_size = self.batch_size
        epoch_size = self.epoch_size
        num_epochs = self.num_epochs
        step_size = self.step_size
        test_split = self.test_split
        
        bsz = batch_size
        tsz = self.time_steps
        isz = input_size
        
        # Make sure that input_size and output_size are nicely divisible by step_size
        assert ( output_size % step_size ) == 0, 'output_size must be divisible by step_size'
        
        ''' MackeyGlass Generator 
        Produces the MackeyGlass dataset for input parameters '''
        def l_mackeyglass( size , inival = 0.2 , del_x = 10 , tao = 30 ):
            mkygls = [ inival ]
            delta = 1 / del_x
            for t in range( int( size ) ):
                if t < tao:
                    yt = mkygls[t] + delta * ((0.2 * mkygls[t])/(1 + pow( mkygls[t] , del_x ) ) - 0.1 * mkygls[t])
                else:
                    yt = mkygls[t] + ((0.2 * mkygls[t-tao])/(1 + pow( mkygls[t-tao] , del_x ) ) - 0.1 * mkygls[t])
                mkygls.append(yt)
            
            return np.asarray( mkygls ).astype( np.float32 )[1::]
            
        '''
        ## PLOTING MACKEY-GLASS DATASET ##
        
        _tau = 20
        _dx = 10
        _y0 = 0.2
        _sz , psz = 1e4 , 1028 # _sz = generating size , psz = plotting size 
        
        rmky1 = l_mackeyglass( _sz , inival = _y0 , del_x = _dx , tao = _tau )
        rmky2 = l_mackeyglass( _sz , inival = _y0 , del_x = _dx , tao = _tau )
        rmky3 = l_mackeyglass( _sz , inival = _y0 , del_x = _dx , tao = _tau )
        rmky4 = l_mackeyglass( _sz , inival = _y0 , del_x = _dx , tao = _tau )
        
        mky1 = rmky1[0:psz]
        mky2 = rmky2[0:psz]
        mky3 = rmky3[0:psz]
        mky4 = rmky4[0:psz]
        x_axs = np.arange( mky1.shape[0] )
        
        filename = 'mkygls_timeseries_'
        filename += 'y_{:0.1f}'.format( _y0 )
        filename += '_tau_' + str( int( _tau ) ) + '_dx_' + str( int( _dx ) ) + '_sz_' + str( int( psz ) )
        filedir = 'documents\\images\\datasets\\mackey_glass'    # From current working directory
        
        title = filename.replace( '_' , ' ' ).capitalize()
        
        do_plot = False
        do_save = True
        do_transparent = True
        format = '.png' 
        _dpi = 1200
#        _figsize = ( 24 , 24 )
        _figsize = ( 9.6 , 7.2 )

        
        ## Image properties ##
        _linwdth = 4
        
        ## Check and build folder for file if it doesn't exist ##
        filedir = os.path.join( os.getcwd() , filedir )
        if not os.path.exists( filedir ): os.makedirs( filedir )
            
        ## Increment file name if one already exists ##
        fncnt = 1
        tmpname = filename
        while os.path.isfile( os.path.join( filedir , tmpname + format ) ):
            tmpname = filename + '_' + str( fncnt )
            fncnt += 1
        filename = tmpname
        
        ## Create Figure & Adjustments ##
        fig = plt.figure( figsize = _figsize )
        ax = fig.add_subplot( )
        #ax.axis('off')
        
        ax.plot( x_axs , mky1 , 'b' , alpha=1.0 )
#        plt.plot( x_axs , mky2 , 'g' , alpha=1.0 )
#        plt.plot( x_axs , mky3 , 'c--' , alpha=1.0 )
#        plt.plot( x_axs , mky4 , 'm--' , alpha=1.0 )
        
        # Print the legend if needed.
#        ax.legend( legend )
        
        # Print textually specific plot features
        plt.title( title )
#        plt.xlabel( x_label )
#        plt.ylabel( y_label )
        
        ## Code to save image ##
        if do_save:
            filename = filename.replace( '.png' , '' )
            filename = os.path.join( filedir , filename + format )
            plt.savefig( filename , dpi=_dpi , transparent=do_transparent , bbox_inches='tight', pad_inches = 0 )
            
            
        if do_plot: plt.show()

        plt.close()
        exit(0)
        '''
        
        cwd = os.getcwd()
        ds_dir = os.path.join( cwd , 'datasets\\np_datasets\\mackey_glass' )
        if not os.path.exists( ds_dir ): os.makedirs( ds_dir )
        
        ## Adds a size and drops the remainders ( 8003 , 10 ) w/ sz=10  -> goes to -> ( 800 , 10 , 10 )
        def expand_size( dat , sz ):
            tssz = sz * ( dat.shape[0] // sz )
            dat = dat[0:tssz,::]
            dat = dat.reshape(( dat.shape[0] // sz , sz , ) + tuple( dat.shape[1::] ))
            return dat
        
        def getmod( a ):
            b = 1e20
            while a // b < 1: b = b // 10
            return int( b )
        
        nmeaddr = '_b' + str( bsz ) + '_t' + str( tsz ) + '_i' + str( isz )
        mky_dirnme = 'mkygls' + nmeaddr + '.npy'
        train_images_dirnme = os.path.join( ds_dir , 'train_images_' + mky_dirnme )
        train_labels_dirnme = os.path.join( ds_dir , 'train_labels_' + mky_dirnme )
        test_images_dirnme = os.path.join( ds_dir , 'test_images_' + mky_dirnme )
        test_labels_dirnme = os.path.join( ds_dir , 'test_labels_' + mky_dirnme )
        if ( os.path.exists( train_images_dirnme ) and os.path.exists( train_labels_dirnme ) and 
             os.path.exists( test_images_dirnme ) and os.path.exists( test_labels_dirnme ) ):
            train_images = np.load( train_images_dirnme , allow_pickle = True )
            train_labels = np.load( train_labels_dirnme , allow_pickle = True )
            test_images = np.load( test_images_dirnme , allow_pickle = True )
            test_labels = np.load( test_labels_dirnme , allow_pickle = True )
        else:
            
            ## Create Data ##
            dsfn = os.path.join( ds_dir , 'mackey_glass_50M.npy' )
            if os.path.exists( dsfn ):
                ds = np.load( dsfn , allow_pickle=True )
                print( "mackey_glass_50M loaded" )
            else:
                ds = l_mackeyglass( 5e6 )
                np.save( dsfn , ds , allow_pickle = True )
                print( "mackey_glass_50M saved" )
                
            self.raw_data = list( ds.flatten() )
            self.raw_size = len( self.raw_data )
            self.raw_shape = [self.raw_size]
            
            # Split into test and train data sets
            splt = ds.shape[0] // 3
            train_data , test_data = ds[0:-splt] , ds[-splt::]
#            print( 'A) train_data.shape:' , train_data.shape )
#            print( 'A) test_data.shape:' , test_data.shape )
            
            # distance to start label vectors -> time-steps over + a vector size + a vector size
            lblskp = tsz + isz
            train_images , train_labels = train_data[0:-lblskp] , train_data[lblskp::]
            test_images , test_labels = test_data[0:-lblskp] , test_data[lblskp::]
#            print( 'A) train_data.shape:' , train_data.shape )
#            print( 'A) test_data.shape:' , test_data.shape )
            
            '''
            xvals = np.arange( train_images.shape[0] )
            plt.plot( xvals[0:512] , train_images[0:512] , 'b' )
            plt.plot( xvals[0:512] , train_labels[0:512] , 'r--' )
            plt.show()
            exit(0)
            #'''
            
            train_images = np.asarray( [ train_images[i:i+isz] for i in range( train_images.shape[0] - isz ) ] )
            train_labels = np.asarray( [ train_labels[i:i+isz] for i in range( train_labels.shape[0] - isz ) ] )
            test_images = np.asarray( [ test_images[i:i+isz] for i in range( test_images.shape[0] - isz ) ] )
            test_labels = np.asarray( [ test_labels[i:i+isz] for i in range( test_labels.shape[0] - isz ) ] )
#            print( 'B) train_images:\n', train_images ,'\nshape:' , train_images.shape , '\n' )
#            print( 'B) train_labels:\n', train_labels ,'\nshape:' , train_labels.shape , '\n' )
#            print( 'B) test_images:\n', test_images ,'\nshape:' , test_images.shape , '\n' )
#            print( 'B) test_labels:\n', test_labels ,'\nshape:' , test_labels.shape , '\n' )
            
            train_images = expand_size( train_images , tsz )
            train_labels = expand_size( train_labels , tsz )
            test_images = expand_size( test_images , tsz )
            test_labels = expand_size( test_labels , tsz )
#            print( 'C) train_images:\n', train_images ,'\nshape:' , train_images.shape , '\n' )
#            print( 'C) train_labels:\n', train_labels ,'\nshape:' , train_labels.shape , '\n' )
#            print( 'C) test_images:\n', test_images ,'\nshape:' , test_images.shape , '\n' )
#            print( 'C) test_labels:\n', test_labels ,'\nshape:' , test_labels.shape , '\n' )
#            print( 'C) train_images.shape:', train_images.shape , '\n' )
#            print( 'C) train_labels.shape:',  train_labels.shape , '\n' )
#            print( 'C) test_images.shape:',  test_images.shape , '\n' )
#            print( 'C) test_labels.shape:',  test_labels.shape , '\n' )
            
            mod = getmod( train_images.shape[0] )
            adj = train_images.shape[0] % mod   # Adjustment to work with tf.model.fit , validation size parameter
            train_images = train_images[0:-adj]
            train_labels = train_labels[0:-adj]
            
            mod = getmod( test_images.shape[0] )
            adj = test_images.shape[0] % mod    # Adjustment to work with tf.model.fit , validation size parameter
            test_images = test_images[0:-adj]
            test_labels = test_labels[0:-adj]

            train_labels = train_labels[::,::,0:1]
            test_labels = test_labels[::,::,0:1]
            
#            print( 'D) train_images:\n', train_images ,'\nshape:' , train_images.shape , '\n' )
#            print( 'D) train_labels:\n', train_labels ,'\nshape:' , train_labels.shape , '\n' )
#            print( 'D) test_images:\n', test_images ,'\nshape:' , test_images.shape , '\n' )
#            print( 'D) test_labels:\n', test_labels ,'\nshape:' , test_labels.shape , '\n' )
            
#            print( 'C) train_images.shape:', train_images.shape , '\n' )
#            print( 'C) train_labels.shape:',  train_labels.shape , '\n' )
#            print( 'C) test_images.shape:',  test_images.shape , '\n' )
#            print( 'C) test_labels.shape:',  test_labels.shape , '\n' )
#            exit(0)
            
            '''
            xvals = np.arange( train_labels.shape[0] )
            plt.plot( xvals[0:512] , train_labels.flatten()[0:512] , 'r--' )
            plt.show()
            exit(0)
            #'''
            
            np.save( train_images_dirnme , train_images , allow_pickle = True )
            np.save( train_labels_dirnme , train_labels , allow_pickle = True )
            np.save( test_images_dirnme , test_images , allow_pickle = True )
            np.save( test_labels_dirnme , test_labels , allow_pickle = True )
            print( 'Saved Training Images: ' + train_images_dirnme )
            print( 'Saved Training Labels: ' + train_labels_dirnme )
            print( '    Saved Test Images: ' + test_images_dirnme )
            print( '    Saved Test Labels: ' + test_labels_dirnme )
            
        trnsz , tstsz = int( 1e4 ) , int( 1e3 * self.batch_size )
        train_images , train_labels = train_images[0:trnsz] , train_labels[0:trnsz]
        test_images , test_labels = test_images[0:tstsz] , test_labels[0:tstsz]
#        print( 'train_images.shape:', train_images.shape , '\n' )
#        print( 'train_labels.shape:',  train_labels.shape , '\n' )
#        print( 'test_images.shape:',  test_images.shape , '\n' )
#        print( 'test_labels.shape:',  test_labels.shape , '\n' )
#        exit(0)
        
        if self.use_dataset:
            
            print( '\n - - - - Mackey Glass Datasets are Activated - - - - \n\n'*3 )
            
            valsz = int( trnsz * self.valid_split )
            
            trnsavedir = os.path.join( os.getcwd() , 'datasets\\tf_datasets\\mackey_glass\\train' + nmeaddr )
            valsavedir = os.path.join( os.getcwd() , 'datasets\\tf_datasets\\mackey_glass\\valid' + nmeaddr )
            tstsavedir = os.path.join( os.getcwd() , 'datasets\\tf_datasets\\mackey_glass\\test' + nmeaddr )
            if self.use_dataset and os.path.exists( trnsavedir ) and os.path.exists( valsavedir ) and os.path.exists( tstsavedir ):
                
                self.train_dataset = tf.data.Dataset.load( trnsavedir )
                self.valid_dataset = tf.data.Dataset.load( valsavedir )
                self.test_dataset = tf.data.Dataset.load( tstsavedir )
                print('Loading datasets: ' + trnsavedir )
                print('Loading datasets: ' + valsavedir )
                print('Loading datasets: ' + tstsavedir , '\n' )
                
            else:
                
                self.train_dataset = tf.data.Dataset.from_tensor_slices( ( train_images[0:-valsz].copy() , train_labels[0:-valsz].copy() ) )
                self.valid_dataset = tf.data.Dataset.from_tensor_slices( ( train_images[-valsz::].copy() , train_labels[-valsz::].copy() ) )
                self.test_dataset  = tf.data.Dataset.from_tensor_slices( ( test_images , test_labels ) )
                
                self.train_dataset = self.train_dataset.batch( self.batch_size ).prefetch( tf.data.AUTOTUNE )
                self.valid_dataset = self.valid_dataset.batch( self.batch_size ).prefetch( tf.data.AUTOTUNE )
                self.test_dataset = self.test_dataset.batch( self.batch_size ).prefetch( tf.data.AUTOTUNE )
            
                self.train_dataset.save( trnsavedir )
                self.valid_dataset.save( valsavedir )
                self.test_dataset.save( tstsavedir )
                
#                self.train_dataset = self.train_dataset.take( -1 )
#                self.valid_dataset = self.valid_dataset.take( -1 )
#                self.test_dataset = self.test_dataset.take( -1 )
                
        # Set the ground truth for image generaton #
        self.gndtru = test_labels.copy()
        
        super().set_data( train_images , train_labels , test_images , test_labels )
        
        return self
        
        '''
        ## Timesteps shift by input_size and input shape is ( None , timesteps , input_size ) ##
        
        timesteps = 25
        
        # Split training , testing , and validation #
        tst_sz = int( ds.shape[0] * self.test_split )
        train_data = ds[0:-tst_sz]
        test_data = ds[-tst_sz::]
        
        # Spit training into images and labels #
        train_images , train_labels = train_data[0:-1] , train_data[1::]
        train_images = expand_size( train_images , timesteps )
        train_labels = expand_size( train_labels , timesteps )
        adj = train_images.shape[0] % 100   # Adjusted to work with tf.model.fit , validation size parameter
        train_images = train_images[0:-adj]
        train_labels = train_labels[0:-adj]
        print( 'train_images shape:' , train_images.shape , '\n' )
        print( 'train_labels shape:' , train_labels.shape , '\n' )
#        exit(0)
        
        # Spit testing into images and labels #
        test_images , test_labels = test_data[0:-1] , test_data[1::]
        test_images = expand_size( test_images , timesteps )
        test_labels = expand_size( test_labels , timesteps )
        adj = test_images.shape[0] % 100    # Adjusted to work with tf.model.fit , validation size parameter
        test_images = test_images[0:-adj]
        test_labels = test_labels[0:-adj]
        print( 'test_images shape:' , test_images.shape , '\n' )
        print( 'test_labels shape:' , test_labels.shape , '\n' )
        
        # Set the ground truth for image generaton #
        self.gndtru = test_labels.copy()
        
        super().set_data( train_images , train_labels , test_images , test_labels )
        
        return self
        #'''
        
    ''' MackeyGlass Generator '''
    # Takes in a set of predictions and outputs,
    #   a 2-tuple (ground truth array, predition array), where the two 
    #   arrays have the same shape. (used for plotting in the trainer save() function)
    def groundtruth_generator( self , preds , maxlen=-1 ):
        
        preds = preds.copy()
        
        if not isinstance( preds , np.ndarray ): preds = np.asarray( preds )
        if not isinstance( self.gndtru , np.ndarray ): self.gndtru = np.asarray( self.gndtru )
        
        # Equalize the predition #
        mn = min( preds.shape[0] , self.gndtru.shape[0] )
        preds , gndtru = preds[0:mn] , self.gndtru[0:mn]
        
        # Make Flat #
        preds = preds.flatten()
        gndtru = gndtru.flatten()
        
        # Make random range & adjust for plottability #
        strt = np.random.randint( 0 , preds.shape[0] - maxlen - 1 )
        preds = preds[strt:strt+maxlen]
        gndtru = gndtru[strt:strt+maxlen]
        
        return ( gndtru , preds )

    ''' DataGenerator '''
    def get_attributes(self,attr={}):
        attr['tao'] =self.tao
        attr['delta_x'] =self.delta_x
        attr['step_size'] =self.step_size
        return super().get_attributes( attr=attr )


'''
Dataset Source:
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. 
    Neural computation, 9(8):1735–1780, 1997.
    <https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext>
'''
class CopyMemoryGenerator(DataGenerator):
    
    ''' CopyMemory Generator '''
    def __init__( 
        self,
        in_bits = 10,
        out_bits = 8,
        low_tol = 1.e-3,
        high_tol = 1.0,
        min_seq = 1,
        max_seq = 20,
        pad = 1.e-3,
        dtype = np.float32,
        name = 'cpymem',
        **kwargs
     ):
     
        super().__init__( **kwargs )
        
        self.raw_size = None
        self.raw_data = None
        
        self.in_bits = in_bits
        self.out_bits = out_bits
        self.low_tol = low_tol
        self.high_tol = high_tol
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.pad = pad
        self.dtype = dtype
        
        
    def gen_cpymem_data( self , time_steps , n_data , n_sequence ):
        ''' SOURCE:
        [1] M. Arjovsky, A. Shah, and Y. Bengio, “Unitary Evolution Recurrent Neural Networks,” p. 9, 2016.
        <https://github.com/stwisdom/urnn/blob/master/memory_problem.py> or <https://github.com/amarshah/complex_RNN/blob/master/memory_problem.py>
        '''
        
        seq = np.random.randint(1, high=9, size=( n_data , n_sequence ))
        zeros1 = np.zeros((n_data, time_steps-1))
        zeros2 = np.zeros((n_data, time_steps))
        marker = 9 * np.ones((n_data, 1))
        zeros3 = np.zeros((n_data, n_sequence))

        x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
        y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
        
        return x.T, y.T
        
        
    ''' CopyMemory Generator '''
    def generate( self ):
        
        self.raw_size = self.epoch_size * self.batch_size if self.raw_size is None else self.raw_size
        
        # --- Set data params ---------------- #
        time_steps = self.max_seq
        n_sequence = 10
        n_input = 10
#        n_output = 9
        n_train = int( 1e5 )
        n_test = int( 25e3 )
#        n_batch = int( n_train / self.batch_size )
        
        # --- Create data --------------------
        train_x, train_y = self.gen_cpymem_data( time_steps , n_train, n_sequence )
        test_x, test_y = self.gen_cpymem_data( time_steps , n_test, n_sequence )
#        print( 'train_x:\n' , train_x , '\nshape:' , train_x.shape , '\n' )
#        print( 'train_y:\n' , train_y , '\nshape:' , train_y.shape , '\n' )
#        exit(0)
        
        train_images = train_x.transpose()
        train_labels = train_y.transpose()
        test_images = test_x.transpose()
        test_labels = test_y.transpose()
        
        train_images = train_images.reshape( train_images.shape[0] // n_input , n_input , train_images.shape[-1] )
        train_labels = train_labels.reshape( train_labels.shape[0] // n_input , n_input , train_labels.shape[-1] )
        test_images = test_images.reshape( test_images.shape[0] // n_input , n_input , test_images.shape[-1] )
        test_labels = test_labels.reshape( test_labels.shape[0] // n_input , n_input , test_labels.shape[-1] )
        
        # Normalize due to Hopf input being touchy with large numbers #
        train_images = train_images / 9.
        train_labels = train_labels
        test_images = test_images / 9.
        test_labels = test_labels
        
        train_images = train_images.astype( self.dtype )
        train_labels = train_labels.astype( self.dtype )
        test_images = test_images.astype( self.dtype )
        test_labels = test_labels.astype( self.dtype )
        
        '''
        ## Uncomment to switch timesteps and image vector ##
        train_images = train_images.transpose(( 0 , 2 , 1 ))
        train_labels = train_labels.transpose(( 0 , 2 , 1 ))
        test_images = test_images.transpose(( 0 , 2 , 1 ))
        test_labels = test_labels.transpose(( 0 , 2 , 1 ))
        #'''
        
        '''
        ## Show train and test shapes ##
        print('train_images -> shape:',train_images.shape,' dtype:',train_images.dtype,'\n')
        print('train_labels -> shape:',train_labels.shape,' dtype:',train_labels.dtype,'\n')
        print('test_images -> shape:',test_images.shape,' dtype:',test_images.dtype,'\n')
        print('test_labels -> shape:',test_labels.shape,' dtype:',test_labels.dtype,'\n')
        exit(0)
        #'''
        
        '''
        ## Plot example image and label ##
        
        seq = self.input_size
        inpt = train_images[1]
        oupt = train_labels[1]
        
        fig , axs = plt.subplots( 2 , 1 )
        idx = 5
        fig.suptitle('Copy Memory Task (Max Sequence Length '+str(seq)+')')
        fig.supxlabel('Time ---->')
        img0 = axs[0].imshow( inpt )
        axs[0].set_ylabel('Image')
        axs[0].set_aspect('auto')
        
        axs[1].imshow( oupt )
        axs[1].set_ylabel('Label')
        axs[1].set_aspect('auto')
        
        fig.colorbar( img0 , ax = axs , orientation='vertical' , fraction=0.1 )
        
        plt.show()
        
        exit(0)
#       '''
        
        trnsz , valsz , tstsz = int( train_images.shape[0] ) , int( self.valid_split * train_images.shape[0] ) , test_images.shape[0]
        nmeaddr = '_seq' + str( time_steps + 20 ) + '_trn' + str( train_images.shape[0] - valsz ) 
        nmeaddr += '_val' + str( valsz ) + '_tst' + str(tstsz )
        if self.use_dataset:
            
            print( '\n - - - - Copy Memory Datasets is Activated - - - - \n\n'*3 )
            
            trnsavedir = os.path.join( os.getcwd() , 'datasets\\tf_datasets\\copy_memory\\train' + nmeaddr )
            valsavedir = os.path.join( os.getcwd() , 'datasets\\tf_datasets\\copy_memory\\valid' + nmeaddr )
            tstsavedir = os.path.join( os.getcwd() , 'datasets\\tf_datasets\\copy_memory\\test' + nmeaddr )
            if self.use_dataset and os.path.exists( trnsavedir ) and os.path.exists( valsavedir ) and os.path.exists( tstsavedir ):
                self.train_dataset = tf.data.Dataset.load( trnsavedir )
                self.valid_dataset = tf.data.Dataset.load( valsavedir )
                self.test_dataset = tf.data.Dataset.load( tstsavedir )
                print('Loading datasets: ' + trnsavedir )
                print('Loading datasets: ' + valsavedir )
                print('Loading datasets: ' + tstsavedir , '\n' )
            else:
                self.train_dataset = tf.data.Dataset.from_tensor_slices( ( train_images[0:-valsz].copy() , train_labels[0:-valsz].copy() ) )
                self.valid_dataset = tf.data.Dataset.from_tensor_slices( ( train_images[-valsz::].copy() , train_labels[-valsz::].copy() ) )
                self.test_dataset  = tf.data.Dataset.from_tensor_slices( ( test_images , test_labels ) )
                
                self.train_dataset = self.train_dataset.batch( self.batch_size ).prefetch( tf.data.AUTOTUNE )
                self.valid_dataset = self.valid_dataset.batch( self.batch_size ).prefetch( tf.data.AUTOTUNE )
                self.test_dataset = self.test_dataset.batch( self.batch_size ).prefetch( tf.data.AUTOTUNE )
                
                self.train_dataset.save( trnsavedir )
                self.valid_dataset.save( valsavedir )
                self.test_dataset.save( tstsavedir )
                print( 'Copy Memory Saved: ' + trnsavedir )
                print( 'Copy Memory Saved: ' + valsavedir )
                print( 'Copy Memory Saved: ' + tstsavedir )
                
        
        self.gndtru = test_labels.copy()
        
        self.set_data( train_images , train_labels , test_images , test_labels )
        
        return self


    ''' CopyMemory Generator '''
    # Takes in a set of predictions and outputs,
    #   a 2-tuple (ground truth array, predition array), where the two 
    #   arrays have the same shape. (used for plotting in the trainer save() function)
    def groundtruth_generator( self , preds , maxlen = -1 ):
        
        preds = preds.copy()
        
        if not isinstance( preds , np.ndarray ): preds = np.asarray( preds )
        
        gndtru = self.gndtru
        if not isinstance( gndtru , np.ndarray ): gndtru = np.asarray( gndtru )
        
        return ( gndtru , preds )

    ''' CopyMemory Generator '''
    def reshape(self):
        print('\nCopyMemoryGeneratorV2|copy: NOT YET IMPLEMENTED!\n')
        return self






''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    