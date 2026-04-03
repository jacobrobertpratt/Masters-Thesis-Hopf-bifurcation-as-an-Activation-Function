import os
import threading

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.keras import backend
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import HopfActCpx , CpxReLU , ModReLU , CpxCard , SigLog
from initializers import Eye , Orthogonal , GlorotNorm , GlorotUnif , Hermitian , Unitary
    

@tf.custom_gradient
def _pgrad( inpt ):
    out = inpt
    def grad( gL ):
        _print( 'gL' , gL )
        return gL
    return out , grad
    

@tf.keras.utils.register_keras_serializable( 'hopf_theta_cell' )
class HopfRNNCellTheta( tf.keras.layers.Layer ):
    '''
    PrattRNNCell processes one timestep of a Hopf-Bifurcation recurrent network.
    That is the input is mapped with the internal weights, and updated 
    '''
    
    def __init__(
            self,
            units,
            activation = 'm2',
            recurrent_weight = 'O',
            input_weight = 'GU',
            train_weights = True,
            save_weights = True,
            **kwargs
        ):
        
        super( HopfRNNCellTheta , self ).__init__( **kwargs )
        
        self.units = units
        self.train_weights = train_weights
        self.save_weights = save_weights
        
        self.actnme = activation
        self.rwgtnme = recurrent_weight
        self.iwgtnme = input_weight
        
        self.recwgt = None
        self.inpwgt = None
        
        self.state_size = units
        self.output_size = units
        
        self.actinpt_size = 1
        self.activate = activation
        self.hopfact = HopfActCpx( units = self.units )
        
        if isinstance( activation , str ):
            tstact = activation.lower()
            if 'm1' == tstact:
                self.activate = self.hopf_config_1
            elif 'm2' == tstact:
                self.activate = self.hopf_config_2
            elif 'm3' == tstact:
                self.activate = self.hopf_config_3
            elif 'm4' == tstact:
                self.activate = self.hopf_config_4
            elif 'm5' == tstact:
                self.activate = self.hopf_config_5
            elif 'm6' == tstact:
                self.activate = self.hopf_config_6
                self.actinpt_size = 2
            elif 'm7' == tstact:
                self.activate = self.hopf_config_7
                self.actinpt_size = 2
            elif 't' == tstact: self.activate = tf.keras.activations.tanh
            elif 'cr' == tstact: self.activate = CpxReLU()
            elif 'cc' == tstact: self.activate = CpxCard()
            elif 'mr' == tstact: self.activate = ModReLU()
            elif 'sl' == tstact: self.activate = SigLog()
            else: self.activate = None
        
        if isinstance( recurrent_weight , str ):
            tstrwgt = recurrent_weight.lower()
            if 'h' == tstrwgt: self.recwgt = Hermitian
            elif 'u' == tstrwgt: self.recwgt = Unitary
            elif 'o' == tstrwgt: self.recwgt = Orthogonal
            elif 'gn' == tstrwgt: self.recwgt = GlorotNorm
            elif 'gu' == tstrwgt: self.recwgt = GlorotUnif
            elif 'i' == tstrwgt: self.recwgt = Eye
            
        if isinstance( input_weight , str ):
            tstiwgt = input_weight.lower()
            if 'h' == tstiwgt: self.inpwgt = Hermitian
            elif 'u' == tstiwgt: self.inpwgt = Unitary
            elif 'o' == tstiwgt: self.inpwgt = Orthogonal
            elif 'gn' == tstiwgt: self.inpwgt = GlorotNorm
            elif 'gu' == tstiwgt: self.inpwgt = GlorotUnif
            elif 'i' == tstiwgt: self.inpwgt = Eye
            
    def b3( self , v ):
        return -tf.math.conj( tf.math.pow( v , 1. / self.units ) )
        
    ## FAILS ##
    def hopf_config_1( self , v ):
        return self.hopfact( v , self.b1 , self.b2 )
    
    def hopf_config_2( self , v ):
        return self.hopfact( self.b1 , v , self.b2 )
    
    def hopf_config_3( self , v ):
        return self.hopfact( self.b1 , self.b1 , self.b3( v ) )
    
    def hopf_config_4( self , v ):
        return self.hopfact( self.b1 , v , self.b3( v ) )
    
    ## FAILS ##
    def hopf_config_5( self , v ):
        return self.hopfact( v , self.b1 , self.b3( v ) )
    
    def hopf_config_6( self , v1 , v2 ):
        return self.hopfact( v1 , v2 , self.b2 )
    
    def hopf_config_7( self , v1 , v2 ):
        return self.hopfact( v1 , v2 , self.b3( v2 ) )
    
    
    def build( self , input_shape ):
        
        '''
        print( self.name + ' units:' , self.units )
        print( self.name + ' state_size:' , self.state_size )
        print( self.name + ' output_size:' , self.output_size )
        print( self.name + ' input_shape:' , input_shape )
        exit(0)
        #'''
        
        inshp = list( input_shape )
        self.bsz = inshp[0]
        self.isz = inshp[-1]
        
        do_save = self.save_weights
        do_train = self.train_weights
        
        wgtshp = ( self.units , self.units )
        
        bshp = ( self.bsz , self.units )
        self.b1 = tf.constant( tf.ones( shape = bshp , dtype = self.dtype ) , dtype = self.dtype )
        self.b2 = tf.constant( (-1.+0.j)*tf.ones( shape = bshp , dtype = self.dtype ) , dtype = self.dtype )
        
        wgtnme = 'u' + str( self.units ) + '_' + self.name.split('_')[0]
        self.A = self.add_weight(
            shape = wgtshp,
            initializer = self.recwgt( name = self.rwgtnme + '_' + wgtnme , save = do_save ),
            trainable = do_train,
            dtype = self.dtype,
            name = 'A_' + wgtnme
        )
        
        self.B = self.add_weight(
            shape = wgtshp,
            initializer = self.inpwgt( name = self.iwgtnme + '_' + wgtnme , save = do_save ),
            trainable = do_train,
            dtype = self.dtype,
            name = 'B_' + wgtnme
        )
        
        '''
        self.gate = self.add_weight(
            shape = ( self.bsz , self.units ),
            initializer = tf.ones,
            trainable = do_train,
            dtype = tf.complex64,
            name = 'gate_' + wgtnme
        )
        '''
        
#        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellTheta , self ).build( input_shape )
        
        self.built = True
        
#    @tf.function
    def split_input( self , v ):
        vstk = tf.unstack( v , axis = -1 )
        return tf.expand_dims( vstk[0] , -1 ) , tf.stack( vstk[1::] , axis = -1 )
        
#    @tf.function
    def combine_output( self , v0 , v_ ):
        vlst = [ tf.squeeze( v0 , -1 ) ] + tf.unstack( v_ , axis = -1 )
        return tf.stack( vlst , axis = -1 )
        
#    @tf.function
    def std_map( self , A , B , v1 , v2 ):
        Av1 = tf.linalg.matmul( v1 , A )
        Bv2 = tf.linalg.matmul( v2 , B )
        return Av1 + Bv2 , Av1 , Bv2
        
#    @tf.function
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        z = states[0] if tf.nest.is_nested( states ) else states
        x = inputs[0] if tf.nest.is_nested( inputs ) else inputs
#        _print( self.name + ' z' , z )
#        _print( self.name + ' x' , x )
        
        z0 , z_ = self.split_input( z )
        x0 , x_ = self.split_input( x )
#        _print( self.name + ' z_' , z_ , summarize=-1 )
        
        z_i , Az , Bx = self.std_map( self.A , self.B , z_ , x_ )
        
        if self.activate is not None:
            y_k = self.activate( z_ , z_i ) if self.actinpt_size == 2 else self.activate( z_i )
        else:
            y_k = z_i
        
        re_j = ( z0 + x0 ) / 2.
        y_t = self.combine_output( re_j , y_k )
        
        z_k , _ = tf.linalg.normalize( z_ + z_i , ord = 2 )
        z_t = self.combine_output( re_j , z_k )
        
        z_t = [ z_t ] if tf.nest.is_nested( states ) else z_t
#        _print( 'y_t' , y_t )
#        _print( 'z_t' , z_t )
        
        return y_t , z_t
        
#        tf.debugging.check_numerics( tf.math.abs( z_t ) , 'var: z_t  ->  hopf cell has nan value' )
#        self.step.assign( tf.math.floormod( self.step + 1 ,  ( self.units + 1 ) ) )
        # Extra Stuff ... #
#        tf.debugging.check_numerics( tf.math.abs( z_t ) , 'var: z_t  ->  hopf cell has nan value' )
        
        
@tf.keras.utils.register_keras_serializable( 'hopf_theta_layer' )
class HopfRNNLayerTheta( tf.keras.layers.Layer ):
    
    def __init__(
        self,
        units,
        output_size = None,
        activation = 'm2',
        recurrent_weight = 'O',
        input_weight = 'GU',
        return_sequences = False,
        time_major = False,
        stateful = False,
        train_weights = True,
        save_weights = True,
        **kwargs
    ):
        
        super( HopfRNNLayerTheta , self ).__init__( **kwargs )
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        self.units = units if units > 3 else 4
        self.osz = output_size if output_size is not None else units
        
        self.activation = activation
        self.recurrent_weight = recurrent_weight
        self.input_weight = input_weight
        
        self.time_major = time_major
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.train_weights = train_weights
        self.save_weights = save_weights
        
        
    def build( self , input_shape ):
        
        self.usz = self.units + 1
        
        inshp = list( input_shape )
        self.bsz = inshp[0]
        self.isz = inshp[1]
        self.vsz = inshp[2] if len( inshp ) == 3 else -1
        
        # Calc. max rfft mapping length -> depends on units and window size
        #rfft_vsz = self.vsz // 2 + 1
        #maxseq = max( self.units , rfft_vsz )
        
        # Add 1 if same as cell units -> we want to be just a bit larger to avoid last column imaginary zeros
        #if maxseq == self.units: maxseq = maxseq + 1
        
        self.seqlen = self.vsz * 2 # ( maxseq - 1 ) * 2 + 2   # x4 instead of x2 to extend input rfft
        
        ## Used to normalize the input rfft mapping ##
        fftnrm = tf.cast( tf.math.sqrt( 0. + self.seqlen ) , dtype = self.dtype )  # Fails when using the Hopf-Activation
#        fftnrm = tf.cast( self.seqlen / 2. , dtype = self.dtype )
        self.fftnorm = tf.constant( fftnrm , dtype = self.dtype )
#        _print( 'fftnrm' , fftnrm )
#        exit(0)
        
        # Create internal hopf cell (represents a single timestep )
        if self.cell is None:
            self.cell = HopfRNNCellTheta(
                units = self.units,
                activation = self.activation,
                recurrent_weight = self.recurrent_weight,
                input_weight = self.input_weight,
                train_weights = self.train_weights,
                save_weights = self.save_weights,
                dtype = self.dtype,
                name = self.name + '_cell'
            )
            
        # From tf.keras.layers.basernn -> calculate cell input shape
        def get_step_input_shape( shape ):
            if isinstance( shape , tf.TensorShape ):
                shape = tuple( shape.as_list() )
            return ( shape[0] , ) + shape[2:]
        step_input_shape = tf.nest.map_structure( get_step_input_shape , input_shape )
        
        # Build cell
        if not self.cell.built:
            with tf.name_scope( self.cell.name ):
                self.cell.build( step_input_shape )         # Last build shape -> self.cells.build( input_shape[1::] )
                assert self.cell.built , 'HopfBifurCpxRNNLayer failed to build.'
                
        state_init = tf.ones( shape = ( self.bsz , self.usz ) , dtype = self.dtype )
        self.state = tf.Variable(
            initial_value = state_init,
            shape = state_init.shape,
            dtype = self.dtype,
            trainable = False,
            name = 'state_'+self.name
        )
        
        super( HopfRNNLayerTheta , self ).build( input_shape )
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
        
#    @tf.function
    def fft_input( self , val ):
        return tf.signal.rfft( val , fft_length = [ self.seqlen ] )[::,::,0:self.usz] / self.fftnorm 
        
        
#    @tf.function
    def fft_output( self , val ):
        return tf.signal.irfft( val * self.fftnorm , fft_length = [ self.seqlen ] )[::,0:self.isz,0:self.osz]
        
        
#    @tf.function
    def call( self , inputs , training = False ):
        
        _input = self.fft_input( inputs )
        
        rnn_return = self.rnn_call(
            self.cell,
            _input,
            self.state,
            return_all_outputs = True,    # If True -> seq_out has batch sized output; else, single output
            training = training
        )
        _clast , _cout , _cstate = rnn_return
        
        if self.stateful: self.state.assign( _cstate )
        
        _output = self.fft_output( _cout )
        
        if not self.return_sequences: _output = _output[::,-1,::]
        
        return tf.cast( _output , inputs.dtype )
        
        
    def get_config(self):
        config = super( HopfRNNLayerTheta , self ).get_config()
        config.update( { 'state_size' : self.state_size } )
        return config
        
        
    @classmethod
    def from_config( self , config ):
        return self( **config )



#    @tf.function
    def rnn_call( self , cell , inputs , states , time_major=False , input_length = None , return_all_outputs=False , training=False):
        
        flat_states = tf.nest.flatten( states )
        
        # Get cell's calling function
        cell_call_fn = ( cell.__call__ if callable(cell) else cell.call )
        
        # Callback for backend.rnn() operations.
        def _step(step_inputs, step_states):
            step_states = step_states[0]
            output, new_states = cell_call_fn(step_inputs, step_states, training=training)
            return output, new_states
        
        new_lastouts, new_outputs, new_states = tf.keras.backend.rnn(
            _step,
            inputs,
            flat_states,
            time_major = time_major,
            return_all_outputs=return_all_outputs
        )
        
        _lastout = new_lastouts[0] if tf.nest.is_nested( new_lastouts ) else new_lastouts
#        _print('_lastout',_lastout)
        
        _outputs = new_outputs[0] if tf.nest.is_nested( new_outputs ) else new_outputs
#        _print('_outputs',_outputs)
        
        _states = new_states[0] if tf.nest.is_nested( new_states ) else new_states
#        _print('_states',_states)
        
        return ( _lastout , _outputs , _states )

