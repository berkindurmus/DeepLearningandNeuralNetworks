import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']
  N=x.shape[0]
  F=w.shape[0]
  Houtx=int(((x.shape[2]+2*pad-w.shape[2])/stride)+1)
  
  Houty=int(((x.shape[3]+2*pad-w.shape[3])/stride)+1)

  
  out=np.zeros((N,F,Houtx,Houty))
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  
  for i in range(N):
      x_paded=np.pad(x[i,:,:,:], [(0,0) ,(pad,pad),(pad,pad)], mode='constant', constant_values=0)
      for j in range (Houtx):  
          for k in range(Houty):
              temp=x_paded[np.newaxis,:,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]]*w
              temp=np.sum(temp,axis=3)
              temp=(np.sum(temp,axis=2))
              temp=np.sum(temp,axis=1)
              out[i,:,j,k]=temp+b

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  #print(dout.shape)
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  #dout_pad=np.pad(dout, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N=x.shape[0]
  F=w.shape[0]
  Houtx=int(((x.shape[2]+2*pad-w.shape[2])/stride)+1)
  Houty=int(((x.shape[3]+2*pad-w.shape[3])/stride)+1)
  #print(Houtx)
  #print(Houty)
  C=x.shape[1]
  dw=np.zeros((F,C,w.shape[2],w.shape[3]))
  dx=np.zeros(())
  dx=np.zeros((4,3,5,5))
  dx_pad=np.zeros(xpad.shape)
 
  for i in range(F):
      for g in range(N):
          x_paded=xpad[g,:,:,:]
          #dout_paded=dout_pad[g,:,:,:]
          #dout_paded=dout_pad[g,:,:,:]
          for j in range (Houtx):  
              for k in range(Houty):
                  ac=x_paded[:,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]]*dout[g,i,j,k]
                  #print(dout.shape)
                  ab=dout[g,i,j,k]*w[i,:,:,:]
                  dx_pad[g,:,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]]+=ab
                  #print(ac.shape)
                  #ab=dout_pad[g,:,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]]*w[:,:,:,:]
                  #dx+=dout_paded[i,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]]
                  dw[i,:,:,:]+=ac
                  dx=dx_pad[:,:,pad:pad+x.shape[2],pad:pad+x.shape[3] ]
                  
                  #dw+=x_paded[i,g,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]]*dout[i,g,]
        
  #w_in=np.flip(np.flip(w,axis=0),axis=1)
  #for i in range(F):
      #for g in range(C):
          #for j in range (Houtx):  
              #for k in range(Houty):
                  #print(dout_pad.shape)
                  #print(k)
                  #print(j)
                  #print(dout_pad[:,i,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]].shape)
                  #print(w_in[i,g,:,:].shape)
                  #ab=dout_pad[:,i,j*stride:j*stride+w.shape[2],k*stride:k*stride+w.shape[3]]*w_in[i,g,:,:]
                  #print(ab.shape)
                  #ab=ab.sum(axis=1).sum(axis=1)
                  #dx[:,g,j,k]+=ab
                  #dw[i,:,:,:]+=ac
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
  db=dout.sum(axis=0).sum(axis=1).sum(axis=1)
  #dx=np.zeros((4,3,5,5))
  #dx=dx[:,:,pad:pad+x.shape[2],pad:pad+x.shape[3] ]
  #db=np.zeros(2)
  
  #print(dx)
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  stride=pool_param['stride']
  pool_height=pool_param['pool_height']
  pool_width=pool_param['pool_width']
    
  Houtx=int((x.shape[2]-pool_param['pool_height'])/pool_param['stride'])+1
  Houty=int((x.shape[3]-pool_param['pool_width'])/pool_param['stride'])+1
  out=np.zeros((x.shape[0],x.shape[1],Houtx,Houty))
  #print(Houtx)
  #print(Houty)
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  for i in range (x.shape[0]):
      for j in range (x.shape[1]):
          
          for k in range(Houtx):
              for h in range(Houty):
                  #print(h*stride+pool_width)
                  #print(x[i,j,k*stride:k*stride+pool_height,h*stride:h*stride+pool_width].shape)
                  out[i,j,k,h]=np.max(x[i,j,k*stride:k*stride+pool_height,h*stride:h*stride+pool_width])
              
 

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  Houtx=int((x.shape[2]-pool_param['pool_height'])/pool_param['stride'])+1
  Houty=int((x.shape[3]-pool_param['pool_width'])/pool_param['stride'])+1
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  dx=np.zeros(x.shape)
  for i in range (x.shape[0]):
      for j in range (x.shape[1]):
          
          for k in range(Houtx):
              for h in range(Houty):
                  #print(h*stride+pool_width)
                  #print(x[i,j,k*stride:k*stride+pool_height,h*stride:h*stride+pool_width].shape)
                  index_x,index_y=np.unravel_index(np.argmax(x[i,j,k*stride:k*stride+pool_height,h*stride:h*stride+pool_width]),x[i,j,k*stride:k*stride+pool_height,h*stride:h*stride+pool_width].shape)
                  index_x=k*stride+index_x
                  index_y=h*stride+index_y
                  dx[i,j,index_x,index_y]=dout[i,j,k,h]
                  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  
  """
  
  #eps = bn_param.get('eps', 1e-5)
  #momentum = bn_param.get('momentum', 0.9)
  #N1=x.shape[0]
  #C=x.shape[1]
  #H=x.shape[2]
  #W=x.shape[3]
  #x=np.transpose(x,(0,2,3,1)).reshape((x.shape[0]*x.shape[2]*x.shape[3], x.shape[1]))
  #N, D = x.shape
  #running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  #running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    
  #out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  out,cache=batchnorm_forward(np.transpose(x,(0,2,3,1)).reshape((x.shape[0]*x.shape[2]*x.shape[3], x.shape[1])), gamma, beta, bn_param)
  N1=x.shape[0]
  C=x.shape[1]
  H=x.shape[2]
  W=x.shape[3]
  out=np.transpose(out.reshape((N1,H,W,C)),(0,3,1,2))
  #temp_x=x
  #sample_mean=np.mean(x,axis=0)
  #sample_var=np.var(x,axis=0)
  #running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  #running_var = momentum * running_var + (1 - momentum) * sample_var
  #a=(x-sample_mean)
  #e=sample_var+eps
  #c=np.sqrt(e)
  #b=1/c
  #x_hat=a*b
  #x=(x-sample_mean)/np.sqrt(sample_var+eps)
  #x=x*gamma+beta
  #x=np.transpose(x.reshape((N1,H,W,C)),(0,3,1,2))
  #out=x
  #temp_x=x
  
  
  #cache=a,e,c,b,x_hat,gamma,sample_var,sample_mean,eps,temp_x
  
  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  dx, dgamma, dbeta = None, None, None
  dx, dgamma, dbeta=batchnorm_backward(np.transpose(dout,(0,2,3,1)).reshape((dout.shape[0]*dout.shape[2]*dout.shape[3], dout.shape[1])), cache)
  a,e,c,b,x_hat,gamma,sample_var,sample_mean,eps,x=cache
  N1=dout.shape[0]
  C=dout.shape[1]
  H=dout.shape[2]
  W=dout.shape[3]
  dx=np.transpose(dx.reshape((N1,H,W,C)),(0,3,1,2))
  #dout=np.transpose(dout,(0,2,3,1)).reshape((dout.shape[0]*dout.shape[2]*dout.shape[3], dout.shape[1]))
 
  
  #a,e,c,b,x_hat,gamma,sample_var,sample_mean,eps,x=cache
  #x_hat=np.transpose(x_hat,(0,2,3,1)).reshape((x_hat.shape[0]*x_hat.shape[2]*x_hat.shape[3], x_hat.shape[1]))
  #print(x.shape)
  #N1=x.shape[0]
  #C=x.shape[1]
  #H=x.shape[2]
  #W=x.shape[3]
  #x=np.transpose(x,(0,2,3,1)).reshape((x.shape[0]*x.shape[2]*x.shape[3], x.shape[1]))
  #dbeta=dout.sum(axis=0)
  #dgamma=(dout*x_hat).sum(axis=0)
  #dx_hat=dout*gamma
  #da=(1/np.sqrt(sample_var+eps))*dx_hat
  #db=(x-sample_mean)*dx_hat
  #dc=(-1/(sample_var+eps))*db
  #de=0.5*((1/np.sqrt(sample_var+eps)))*dc
  #dvar=de.sum(axis=0)
  #dmu=-da.sum(axis=0)-dvar*(2/x.shape[0])*((x-sample_mean).sum(axis=0))
  #dx=(1/np.sqrt(sample_var+eps))*dx_hat+(2*(x-sample_mean)/x.shape[0])*dvar+dmu/x.shape[0]
  #dx=np.transpose(dx.reshape(N1,H,W,C),(0,3,1,2))
  #dx=np.transpose(x.reshape((N1,H,W,C)),(0,3,1,2))
 
  #print(dx.shape)
  #batch_backward(dout,cache)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta
def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the batchnorm backward pass, calculating dx, dgamma, and dbeta.
    # ================================================================ #
    a,e,c,b,x_hat,gamma,sample_var,sample_mean,eps,x=cache
    dbeta=dout.sum(axis=0)
    dgamma=(dout*x_hat).sum(axis=0)
    dx_hat=dout*gamma
    da=(1/np.sqrt(sample_var+eps))*dx_hat
    db=(x-sample_mean)*dx_hat
    dc=(-1/(sample_var+eps))*db
    de=0.5*((1/np.sqrt(sample_var+eps)))*dc
    dvar=de.sum(axis=0)
    dmu=-da.sum(axis=0)-dvar*(2/x.shape[0])*((x-sample_mean).sum(axis=0))
    dx=(1/np.sqrt(sample_var+eps))*dx_hat+(2*(x-sample_mean)/x.shape[0])*dvar+dmu/x.shape[0]
    
    
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return dx, dgamma, dbeta
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        
        # ================================================================ #
        # YOUR CODE HERE:
        #   A few steps here:
        #     (1) Calculate the running mean and variance of the minibatch.
        #     (2) Normalize the activations with the running mean and variance.
        #     (3) Scale and shift the normalized activations.  Store this
        #         as the variable 'out'
        #     (4) Store any variables you may need for the backward pass in
        #         the 'cache' variable.
        # ================================================================ #
        temp_x=x
        sample_mean=np.mean(x,axis=0)
        sample_var=np.var(x,axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        a=(x-sample_mean)
        e=sample_var+eps
        c=np.sqrt(e)
        b=1/c
        x_hat=a*b
        x=(x-sample_mean)/np.sqrt(sample_var+eps)
        x=x*gamma+beta
        out=x
        cache=a,e,c,b,x_hat,gamma,sample_var,sample_mean,eps,temp_x
        pass

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
    elif mode == 'test':
        # ================================================================ #
        # YOUR CODE HERE:
        #   Calculate the testing time normalized activation.  Normalize using
        #   the running mean and variance, and then scale and shift appropriately.
        #   Store the output as 'out'.
        # ================================================================ #
        x=(x-running_mean)/np.sqrt(running_var+eps)
        x=x*gamma+beta
        out=x
        pass
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache