''' Special Imports '''
import os
import sys
import json
from datetime import datetime

''' Special Imports '''
# Linear Algebra
import numpy as np
import scipy
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def _print( msg , val ):
    print( msg + ':\n' , val , '\nshape:' , val.shape , '  dtype:' , val.dtype , '  type:' , type( val ) , '\n' )
    
    
def cpx_hopf_step( z , a , b ):
    zz = np.conjugate( z ) * z
    bzz = b * zz
    dz = ( a + bzz ) * z
    return dz
    
    
''' TESTING SECTION '''
if __name__ == "__main__":
    
    ## Cropped 3149
    
#    filename = 'limit_cycle_surface'  # Used for parabolic surface stuff
#    filedir = 'limit_cycle_surface'
    
    filename = 'a_gtr_0_b_gtr_0_plane'
    filedir = ''
    
    do_plot = True
    do_save = True
    do_transparent = True
    format = '.png' 
    _dpi = 600
    _figsize = ( 12 , 12 )
    
    
    ## Image properties ##
    _linwdth = 3
    
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
    ax = fig.add_subplot( projection='3d' )
    ax.axis('off')
    
    sz = 20
    plns = 1
    
    _a = (0.75 + 0.5j)
    _b = (-0.75 + 0.5j)
    
    ## For adjusting planes slices and sizes ##
    slclst = [ ]
    sszlst = [ ]
    xrng = [ 1. ]
    
    ### Check inputs
    xrng = np.asarray( xrng )
    xrng = np.linspace( 1.0 , -1.0 , plns ) if xrng.shape[0] != plns else xrng
    
    slclst = [ ( 0 , 'n' ) ]*plns if len( slclst ) != plns else slclst
    sszlst = [ 1 ]*plns if len( sszlst ) != plns else sszlst
    
    yrng = np.linspace( -2. , 2. , sz )
    zrng = np.linspace( -2. , 2. , sz )
    gx , gy , gz = np.meshgrid( xrng , yrng , zrng )
    
#    _r = np.sqrt( _a.real / _b.real ) * xrng
    
    ## Build Hopf-Quivers ##
    z = gy + 1.j*gz
    a = gx * _a                 # gx here acts like mu
    b = np.ones_like( z ) * _b
    
    # Run one hopf step
    dz = cpx_hopf_step( z , a , b )
    dy , dz = dz.real , dz.imag
    dx = np.zeros_like( gx )
    
    ## Uncomment to plot quiver planes ##
    def slice( g , d , slc ):
        adj = slc[0]
        axs = slc[1]
        sh = list( g.shape )[-1] // 2 + adj
        if axs == 'y':
            return g[sh::,::,::] , d[sh::,::,::]
        elif axs == 'z':
            return g[::,::,sh:-sz] , d[::,::,sh:-sz]
        return g , d
    
    def resz( g , d , sz ):
        return g[sz:-sz,::,sz:-sz] , d[sz:-sz,::,sz:-sz]
    
    def setssz( glst , dlst , ssz ):
        for i in range( len( ssz ) ):
            glst[i] , dlst[i] = resz( glst[i] , dlst[i] , ssz[i] )
        return glst , dlst
    
    def setslc( glst , dlst , slst ):
        for i in range( len( slst ) ):
            glst[i] , dlst[i] = slice( glst[i] , dlst[i] , slst[i] )
        return glst , dlst
        
    assert len( slclst ) == plns , 'Slice list must be same size as number of planes'
    assert len( sszlst ) == plns , 'Resize list must be same size as number of planes'
    
    gz = np.split( gz , plns , 1 )
    dz = np.split( dz , plns , 1 )
    gz , dz = setssz( gz , dz , sszlst )
    gz , dz = setslc( gz , dz , slclst )
    
    gy = np.split( gy , plns , 1 )
    dy = np.split( dy , plns , 1 )
    gy , dy = setssz( gy , dy , sszlst )
    gy , dy = setslc( gy , dy , slclst )
    
    gx = np.split( gx , plns , 1 )
    dx = np.split( dx , plns , 1 )
    gx , dx = setssz( gx , dx , sszlst )
    gx , dx = setslc( gx , dx , slclst )
    
    # Add quivers to graph
    qvrlen = np.linspace( 0.05 , 0.1 , plns )
    for i in range( len( gx ) ):
        ax.scatter( gx[i] , gy[i] , gz[i] , s=0.2 , c='000000' , marker='o' )
        ax.quiver( gx[i] , gy[i] , gz[i] , dx[i] , dy[i] ,dz[i] , length=0.2 , normalize=True , linewidth=_linwdth - 1 , arrow_length_ratio=0.4 )
    
    #'''
    ## Build hopf limit-cycle circles ##
    crng = np.linspace( 0. , 2*np.pi , 128 )
    r = np.sqrt( _a.real / np.abs( _b.real ) )
    cy = r*np.cos( crng )
    cz = r*np.sin( crng )
    cx = np.ones_like( cy )
    ax.plot( cx , cy , cz , 'k' , linewidth = _linwdth )
    #'''
    
    lnsz = 2.5
    zeros = [ 0 , 0 ]
    ax.plot( zeros , [ -lnsz , lnsz ] , zeros , 'k--' , linewidth = _linwdth )
    ax.plot( zeros , zeros , [ -lnsz , lnsz ] , 'k--' , linewidth = _linwdth )
    ax.view_init( elev=0 , azim=0 , roll=0 ) # YZ Axis
    
    ## Code to save image ##
    if do_save:
        filename = filename.replace( '.png' , '' )
        filename = os.path.join( filedir , filename + format )
        plt.savefig( filename , dpi=_dpi , transparent=do_transparent , bbox_inches='tight', pad_inches = 0 )
        
        
    if do_plot: plt.show()
    
    plt.close()