import numpy as np
from scipy.spatial import distance
from itertools import product
from numpy import arctan2

# kernels
"""
X: array of shape(n1,m)
Y: array of shape(n2,m)
"""
def linear_kernel(X,Y):
    X=np.asarray(X)
    Y=np.asarray(Y)
    return np.dot(X,Y.T)

def polynomial_kernel(X,Y,d=3,gamma=1.,c=0.):
    """
    K(X,Y) = (gamma*<X,Y>+c)^d
    """
    X=np.asarray(X)
    Y=np.asarray(Y)
    return (gamma*np.dot(X,Y.T) + c)**d

def gaussian_kernel(X,Y,gamma=1.):
    """
    K(x, y) = exp(-gamma ||X-Y||^2)
    """
    X=np.asarray(X)
    Y=np.asarray(Y)
    return np.exp(-gamma*(distance.cdist(X,Y,'euclidean')**2))

def laplacian_kernel(X,Y,gamma=0.01):
    """
    K(X, Y) = exp(-gamma ||x-y||)
    """
    X=np.asarray(X)
    Y=np.asarray(Y)
    return np.exp(-gamma*(distance.cdist(X,Y,'cityblock')))

 
KERNELS = {
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'gaussian': gaussian_kernel,
    'laplacian': laplacian_kernel}

# distances in the feature space
"""
dk(x1,x2)^2 = K(x1,x1)+K(x2,x2)-2K(x1,x2)
"""
def dist_kernel(X,Y,metric,param=None):
    n_samples_X, _ = X.shape
    n_samples_Y, _ = Y.shape
    if metric in KERNELS:
        K = KERNELS[metric]
        Kx = K(X,X,**param)
        Ky = K(Y,Y,**param)
        Kxy = K(X,Y,**param)
        matrix_dist = [(Kx[i,i]+Ky[j,j]-2*Kxy[i,j]) for i,j in product(range(n_samples_X), range(n_samples_Y))]
        matrix_dist = np.reshape(matrix_dist, (n_samples_X, n_samples_Y))
        return matrix_dist
    else:
        raise ValueError('Unknown kernel  %s' % metric)

def gradientmap_grey_kernel(X):
    r,c = X.shape
    gradient =[]
    for idx in range(r):
        tmp = X[idx]
        R_channel,G_channel,B_channel = tmp[:1024],tmp[1024:2048],tmp[2048:]
        grey = np.mean( np.array([ R_channel, G_channel,B_channel ]), axis=0 )
        grey = np.reshape(grey,(32,32))
        grad_x, grad_y = np.gradient(grey)
        grad_x = grad_x.reshape((32,32,1))
        grad_y = grad_y.reshape((32,32,1))
        gradient.append(np.concatenate((grad_x,grad_y),axis=2))
    aaaa = np.reshape(gradient,(-1,2))
    ss = []
    for i in range(len(aaaa)):
        ss.append(np.arctan(aaaa[i][0]/aaaa[i][1]))
    s = np.reshape(ss,(-1,32,32))
    return s

def averageColor_kernel(X):
    r,c = X.shape
    s =[]
    for idx in range(r):
        tmp = X[idx]
        R_channel,G_channel,B_channel = tmp[:1024],tmp[1024:2048],tmp[2048:]
        s.append(np.mean( np.array([ R_channel, G_channel,B_channel ]), axis=0 ))
    return s    

#### HOG ####
def gray(image):
    """
    Image 3D colored to 2D gray
    
    """
    R, G, B = image[:,:,0],image[:,:,1],image[:,:,2]
    Gray = R*0.3+G*0.59+B*0.11
    Gray = Gray**1/2
    
    #image = [Gray,Gray,Gray]
    image = np.reshape(Gray,(32,32))
    return image

def gradient(image):
    #the gradients will have shape (sy-2, sx-2)
    sy, sx = image.shape
    gx = np.zeros((sy-2, sx-2))
    gx[:, :] = -image[1:-1, :-2] + image[1:-1, 2:]

    gy = np.zeros((sy-2, sx-2))
    gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]
    
    return gx, gy

def magnitude_orientation(gx, gy):
    # element-wise
    gx, gy = np.array(gx), np.array(gy)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (arctan2(gy, gx) * 180 / np.pi) % 360
            
    return magnitude, orientation

def nbins(magnitude, orientation, nb):
    max_angle = 360
    #orientation =np.array(orientation)

    b_step = max_angle/nb
    b0 = (orientation % max_angle) // b_step   #[30x30]
    
    b0[np.where(b0>=nb)]=0

    temp_coefs = np.zeros(nb)
    for m, o  in zip(magnitude, b0):
        temp_coefs[o] += m*1e2
    
    return temp_coefs
    
def compute_cells(magnitude,orientation,npixel, nb):
    # npixel: number of pixels in a cell
    sy, sx = orientation.shape
    
    cells_o = [orientation[i:i+npixel,j:j+npixel] for i in range(sy-npixel+1) for j in range(sx-npixel+1)] 
    cells_m = [magnitude[i:i+npixel,j:j+npixel] for i in range(sy-npixel+1) for j in range(sx-npixel+1)] 
    
    hist = [nbins(np.ravel(cells_m[k]),np.ravel(cells_o[k]),nb) for k in range(len(cells_o))]
    
    return hist

def feature_extraction_per_image(image, npixel, nb):
    image = gray(image)
    gx, gy = gradient(image)
    magnitude, orientation = magnitude_orientation(gx,gy)
    hist = compute_cells(magnitude,orientation,npixel, nb)
    return np.ravel(hist)

def batchmap(X):
    r,c = X.shape
    batch = []
    for idx in range(r):
        tmp = X[idx]
        R_channel,G_channel,B_channel = tmp[:1024],tmp[1024:2048],tmp[2048:]
        R,G,B = np.reshape(R_channel,(32,32,1)),np.reshape(G_channel,(32,32,1)),np.reshape(B_channel,(32,32,1))
        batch.append(np.concatenate((R,G,B),axis=2))
    return batch  #size of batchmap: (3072L, 32L, 32L, 3L)

def feature_hist(image, npixel, nb):
    batch = batchmap(image)
    nimg,_,_,_ = np.shape(batch)

    hist_data = [(feature_extraction_per_image(batch[i], npixel, nb)) for i in range(nimg)] 
    
    return np.array(hist_data)

def hist_kernel(X,Y,npixel=6,nb=9):
	return linear_kernel(feature_hist(X,npixel,nb),feature_hist(Y,npixel,nb))
