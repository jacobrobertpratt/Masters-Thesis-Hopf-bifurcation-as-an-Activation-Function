''' Standard Imports '''
import os
import sys
import math
import random
from datetime import datetime


''' Special Imports '''
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.gridspec as gridspec

import scipy
from scipy.integrate import odeint, solve_ivp


_linwdth = 1
_fontsize = 10
_mkrsize = 1    


## Step function for scipy.odeint ##
def cpx_hopf_func( z , t , re_a , im_a , re_b , im_b ):
    
    z = z[0] + 1.j*z[1]
    a = re_a + 1.j*im_a
    b = re_b + 1.j*im_b
    
    zz = np.conjugate( z ) * z
    bzz = b * zz
    dz = ( a + bzz ) * z
    
    return [ dz.real , dz.imag ]


## Plots one ODE Solution ##
def plot_ode_sol( _ax , _stepfunc , _state , _time , _odeargs ):
    ode = odeint( _stepfunc , _state , _time , args = _odeargs ).T
    _ax.plot( _time , ode[0] , ode[1] , linewidth=_linwdth , alpha=1.0 , color='firebrick' )


def tickmarks( _ax , stp , cnt , axis='x', label=False, **kwargs ):
    
    if 'marker' not in kwargs: kwargs['marker'] = '+'
    if 'c' not in kwargs: kwargs['c'] = 'k'
    
    if axis == 'x':
        for n in range( int( cnt ) ): ax.scatter( stp*n , 0 , 0 , s=_mkrsize , linewidth=_linwdth , **kwargs )
    elif axis == 'y':
        for n in range( int( cnt ) ): ax.scatter( 0 , stp*n , 0 ,  s=_mkrsize , linewidth=_linwdth , **kwargs )
    elif axis == 'z':
        for n in range( int( cnt ) ): ax.scatter( 0 , 0 , stp*n , s=_mkrsize , linewidth=_linwdth , **kwargs )
    else:
        pass
        
    if label is True:
        _zdir = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
        tcnt = cnt - 1
        minor_marks = np.round( np.linspace( stp , stp*tcnt , tcnt ) , 2 )
        if axis == 'x':
            for mark in minor_marks: ax.text( mark - 0.15 , 0 , -0.15 , str( mark ) , fontsize=_fontsize )
        elif axis == 'y':
            for mark in minor_marks: ax.text( 0 , mark , 0 , str( mark ) , fontsize=_fontsize )
        elif axis == 'z':
            for mark in minor_marks: ax.text( -0.8 , 0 , mark - 0.025 , str( mark ) , 'x' , fontsize=_fontsize )
        else:
            pass
            
            
def get_ax_size(ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height
    
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
if __name__ == "__main__":
    
    do_plot = False
    do_save = True
    do_transparent = True
    format = '.png' 
    _dpi = 1200
    _figsize = ( 6 , 6 )
#    _figsize = ( 10 , 10 )

    ## Image properties ##
    _axlim = [ [ -0.5 , 2*np.pi ] , [0,0] , [-2,2] ]
#    _linwdth = 1
    
    ## ODE Parameters ##
    _time = np.linspace( 0. , 2.*np.pi , 1024 )
    
    cpx = np.exp( -2.j*np.pi/8 )
    state =  [ cpx.real , cpx.imag ]
    
    au = np.linspace(-1.5 , 1.5 , 4 )
    bu = np.linspace(-3.0 , -1.0 , 4 )
    
    aw = np.linspace( np.pi/8 , 2*np.pi , 4 )
    bw = np.linspace( 2*np.pi , np.pi/8 , 4 )
    
    for r in range( len( au ) ):
        for c in range( len( aw ) ):
        
            filename = 'pos_' + str( r ) + '_' + str( c )
            filedir = ''
            
            if do_save:
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
            grdspc = gridspec.GridSpec( 1 , 1 , left = 0 , right=1 , top = 1 , bottom = 0 )
            grdspc.update(  wspace=0.0 , hspace=0.0 )
            
            ax = fig.add_subplot( grdspc[0] , projection='3d' )
            ax.axis('off')
            ax.set_xlim3d([-0.75,7.75])
            ax.set_ylim3d([-1,1])
            ax.set_zlim3d([-0.8,1.5])
            
            re_a = au[ r ]
            re_b = bu[ r ]
            
            im_a = aw[ c ]
            im_b = bw[ c ]
            
            ode = plot_ode_sol( ax , cpx_hopf_func , state , _time , _odeargs = ( re_a , im_a , re_b , im_b ) )
            
            rnd = 3
            zstr = ' = ('+str(round( cpx.real , rnd ) )+'+ '+str(round( cpx.imag , rnd ) )+'j)'
            astr = ' = ('+str( round( re_a , rnd ) )+'+ '+str(round( im_a , rnd ))+'j)'
            bstr = ' = ('+str( round( re_b , rnd ) )+'+ '+str(round( im_b , rnd ))+'j)'
            
            ## Add Text for axis and labels ##
            ax.text( 4 , 0 , 1.7 , zstr , color='k' , fontweight='bold' )
            ax.text( 4 , 0 , 1.55 , astr , color='k' , fontweight='bold' )
            ax.text( 4 , 0 , 1.40 , bstr , color='k' , fontweight='bold' )
            ax.text( -0.6 , 0 , -0.15 , 'Re' , color='k' , fontweight='bold' )
            ax.text( 2*np.pi , 0 , 0.1 , 'time' , color='k' , fontweight='bold' )
            ax.text( -0.6 , 0 , 1.35 , 'Im' , color='k' , fontweight='bold' )
            
            ## Add Lines to graph ##
            zros = [ 0. , 0. ]
            ax.plot([-0.5,2*np.pi], zros, zros , 'k-' , linewidth = _linwdth )
            ax.plot( zros , zros , [ -1.25 , 1.25 ] , 'k-' , linewidth = _linwdth )
            
            tickmarks( ax , 1.5 / 4 , 4 , axis='z' , label=True )
            tickmarks( ax , -1.0 , 2 , axis='z' , label=True )
        #    tickmarks( ax , np.pi/4 , 4 , axis='z' , label=True )
            tickmarks( ax , np.pi/2 , 4 , axis='x' , label=True )
            
            ## Adjust plot to view YZ axis ##
            ax.view_init( elev=0 , azim=-92 , roll=0 ) # YZ Axis View
            
            ## Code to save image ##
            if do_save:
                filename = filename.replace( '.png' , '' )
                filename = os.path.join( filedir , filename + format )
                plt.savefig( filename , dpi=_dpi , transparent=do_transparent , bbox_inches='tight', pad_inches = 0 )
        
            if do_plot: plt.show()
            
        
    plt.close()