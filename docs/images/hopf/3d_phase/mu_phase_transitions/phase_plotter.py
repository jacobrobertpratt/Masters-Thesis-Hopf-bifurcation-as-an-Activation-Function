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
    bzz = (-1.+0.j) * b * zz
    dz = ( a + bzz ) * z
    return dz
    
    
''' TESTING SECTION '''
if __name__ == "__main__":
    
    
#    filename = 'limit_cycle_surface'  # Used for parabolic surface stuff
#    filedir = 'limit_cycle_surface'
    
    filename = 'iso_phase_planes'
    filedir = ''
    
    do_plot = False
    do_save = True
    do_transparent = True
    format = '.png' 
    _dpi = 600
    if do_save:
        _figsize = ( 24 , 24 )
    else:
        _figsize = ( 10 , 10 )
        _figsize = ( 24 , 24 )
    
    ## Image properties ##
    _linwdth = 4
    
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
    plns = 3
    _a = (0.75 + 0.5j)
    _b = (0.75 - 0.0j)
    
    ## For adjusting planes slices and sizes ##
    slclst = [ ]
    sszlst = [ 5 , 4 , 3 ]
#    xrng = [ -1. ]
#    xrng = [ 0. ]
    xrng = [ -1. , 0. , 1. ]
    
    ### Check inputs
    xrng = np.asarray( xrng )
    xrng = np.linspace( 1.0 , -1.0 , plns ) if xrng.shape[0] != plns else xrng
    
    slclst = [ ( 0 , 'n' ) ]*plns if len( slclst ) != plns else slclst
    sszlst = [ 1 ]*plns if len( sszlst ) != plns else sszlst
    
    yrng = np.linspace( -2. , 2. , sz )
    zrng = np.linspace( -2. , 2. , sz )
    gx , gy , gz = np.meshgrid( xrng , yrng , zrng )
    
    _r = np.sqrt( _a.real / _b.real ) * xrng
    
    ## Build Hopf-Quivers ##
    z = gy + 1.j*gz
    a = gx * _a                 # gx here acts like mu
    b = np.ones_like( z ) * _b
    
    # Run one hopf step
    dz = cpx_hopf_step( z , a , b )
    dy , dz = dz.real , dz.imag
    dx = np.zeros_like( gx )
    
    #'''
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
        ax.quiver( gx[i] , gy[i] , gz[i] , dx[i] , dy[i] , dz[i] , length=0.2 , normalize=True , linewidth=_linwdth - 1 , arrow_length_ratio=0.4 )
        
    '''
    lsz = 2.5
    zeros = [0, 0]
    ax.plot( [-lsz, lsz] , zeros , zeros , 'k-' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( zeros , [-lsz,lsz] , zeros , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , zeros , [-lsz,lsz] , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.view_init( elev=0 , azim=0 , roll=0 ) # YZ Axis
    '''
    
    '''
    ## Uncomment to remove all quivers ##
    #ax.quiver( gx , gy , gz , dx , dy , dz , length=0.2 , normalize=True ) # , alpha=0.75 , color='k')
    #'''
        
    #'''
    ## Construct bifurcation hyperbola ##
    sz = 50
    d = np.sqrt( _b.real / _a.real )
    r = np.linspace( 0. , 1. , sz )
    theta = np.linspace( 0. , 2*np.pi , sz )
    r , th = np.meshgrid( r , theta )
    py = r * np.cos( th ) / d
    pz = r * np.sin( th ) / d
    px = (d**2) * ( pz**2 + py**2 )
    
    
    zeros = [0, 0]
    dsh = 0.715
    ax.plot( [-1.5, 0.0] , zeros , zeros , 'k-' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( [ 0.0, dsh ] , zeros , zeros , 'k--' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( [ dsh, 1.75 ] , zeros , zeros , 'k-' , linewidth=_linwdth ) # , alpha=0.5 )
    dsh=0.18
    ax.plot( zeros , [-1.75,-dsh] , zeros , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , [-dsh,1.75] , zeros , 'k--' , linewidth=_linwdth ) #, alpha=0.5 )
    dsh = 0.39
    dsh2 = 0.12
    ax.plot( zeros , zeros , [-1.75,-dsh] , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , zeros , [-dsh,dsh2] , 'k--' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , zeros , [dsh2,1.75] , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.view_init( elev=0 , azim=0 , roll=0 ) # YZ Axis
    
#    ax.plot_surface( px , py , pz , alpha=0.25 , color = '#00ff00') # Green
#    ax.plot_surface( px , py , pz , alpha=0.5 , color = '#0099ff') # Light-Blue
    ax.plot_surface( px , py , pz , alpha=0.25 , color = '#ffff') # gray
    
    ax.view_init( elev=23 , azim=-70 , roll=0 )    # Isometric
    
    #'''
    
    '''
    ## Construct bifurcation parabola ##
    sz = 50
    d = np.sqrt( _b.real / _a.real )
    pz = np.linspace( -1. , 1. , sz )
    py = np.zeros_like( pz )
    px = pz**2
    ax.plot( px , py , pz , 'k' , linewidth=_linwdth )
    
    ## Dashed lines for parabola ##
    zeros = [0, 0]
    ax.plot( [-1.25, 0.] , zeros , zeros , 'k-' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( [ 0., 1.0 ] , zeros , zeros , 'k--' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( [ 1.0, 1.5 ] , zeros , zeros , 'k-' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( zeros , [-1.25,1.25] , zeros , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , zeros , [-1.25,1.25] , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    
    ## Left-side video for parabola ##
    ax.view_init( elev=0 , azim=-88 , roll=0 )     # Left-sde with X & Z axis
    
    #'''
    
    #'''
    ## Build hopf limit-cycle circles ##
    cyrng = np.linspace( 0. , 2*np.pi , 32 )
    czrng = np.linspace( 0. , 2*np.pi , 32 )
    
    circs = np.linspace( 0. , 1. , 2 )
    for c in circs:
        r = np.sqrt( _a.real * c / _b.real )
        cy = np.cos( cyrng )*r
        cz = np.sin( czrng )*r
        cx = np.ones_like( cy )*c
        ax.plot( cx , cy , cz , 'k' , linewidth=_linwdth )   # for circles
    
    #'''
    
    '''
    ## For Circle Dashed Lines ##
    ## Add x,y,z axes to the graph ##
    zeros = [0, 0]
    ax.plot( [-1.25, 0.] , zeros , zeros , 'k-' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( [ 0., 1.0 ] , zeros , zeros , 'k--' , linewidth=_linwdth ) # , alpha=0.5 )
    ax.plot( [ 1.0, 1.5 ] , zeros , zeros , 'k-' , linewidth=_linwdth ) # , alpha=0.5 )
    dsh = 1.05
    ax.plot( zeros , [-1.25,-dsh] , zeros , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , [-dsh,dsh] , zeros , 'k--' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , [dsh,1.25] , zeros , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , zeros , [-1.25,-dsh] , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , zeros , [-dsh,dsh] , 'k--' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.plot( zeros , zeros , [dsh,1.25] , 'k-' , linewidth=_linwdth ) #, alpha=0.5 )
    ax.view_init( elev=0 , azim=0 , roll=0 ) # YZ Axis
    #'''
    
    
    ## Code to save image ##
    if do_save:
        filename = filename.replace( '.png' , '' )
        filename = os.path.join( filedir , filename + format )
        plt.savefig( filename , dpi=_dpi , transparent=do_transparent , bbox_inches='tight', pad_inches = 0 )
        
        
    if do_plot: plt.show()
    
    plt.close()