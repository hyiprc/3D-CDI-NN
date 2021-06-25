from typing import Union
import numpy as np

debug = False
zero = 1e-6

# load model setup
try:
    import json
    with open('nn_model/setup.json') as f:
        setup = json.load(f)
    dimension = setup['dimension']
    data_format = setup['data_format']
except: pass


# ------------------------------------------------------
def load_3d_tiff(fname:str)->np.ndarray:
    """ load multiframe tiff file """
    import tifffile
    # undo reorient based on 3D view in imagej
    tiff2npy = lambda img:np.rot90(img[:,:,::-1],1,axes=(0,2))
    return tiff2npy(tifffile.imread(fname))


def save_3d_tiff(fname:str, img:np.ndarray)->None:
    """ save multiframe tiff file """
    import tifffile
    if img.ndim==2: 
        tifffile.imsave(fname,img.astype('int32'))
        return
    # reorient based on 3D view in imagej
    npy2tiff = lambda img:np.rot90(img[:,:,::-1],1,axes=(0,2))
    tifffile.imsave(fname,npy2tiff(np.abs(img).astype(np.float32)))


# ------------------------------------------------------
def fft(fp:np.ndarray)->np.ndarray:
    return np.fft.fftshift(np.fft.fftn(fp))


def ifft(fp:np.ndarray)->np.ndarray:
    real = np.fft.ifftn(np.fft.ifftshift(fp))
    real[np.abs(real)<zero] = 0
    return real


# ------------------------------------------------------
def phase_wrap(cpx:np.ndarray, threshold=0.3):
    amp, ang = np.abs(cpx), np.angle(cpx)
    avg = np.mean(ang[amp>threshold])
    ang -= avg
    ang = (ang+np.pi) % (2.*np.pi) - np.pi
    return amp * np.exp(1j*ang)


def cpx_real_resize(cpx:np.ndarray, sizes:Union[int,list])->np.ndarray:
    """ Resize Image to target size, scale amp and phase seperately.

        Only use this for scaling in real space,
        use dct_scale for reciprocal space which preserves oversampling ratio.
    """

    from skimage.transform import resize

    size = tuple(np.tile(sizes,3)[:3])
    out = resize(
        np.real(cpx), size, preserve_range=True
    ) + 1j*resize(
        np.imag(cpx), size, preserve_range=True
    )
    norm = np.max(out)
    return out/norm


def crop(img:np.ndarray, n:Union[int,list]=(64,64,64))->np.ndarray:
    """ crop image, pad by zero if larger than image dimension """
    d = img.ndim
    n = np.tile(n,d)[:d]
    n = tuple(np.floor(n).astype(int))
    nn = np.min(np.c_[img.shape,n],axis=1)
    n1 = np.round(0.5*(np.array(img.shape)-nn),0).astype(int)
    n2 = n1+np.round(n).astype(int)
    n3 = np.round(0.5*(np.array(n)-nn),0).astype(int)
    n4 = np.round(n3+nn,0).astype(int)
    outimg = np.zeros(n,dtype=img.dtype)
    outimg[n3[0]:n4[0],n3[1]:n4[1],n3[2]:n4[2]] = \
        img[n1[0]:n2[0],n1[1]:n2[1],n1[2]:n2[2]]
    return outimg


def center_to_com(cpx:np.ndarray)->np.ndarray:
    extent = tuple([slice(np.min(s),np.max(s)) for s in np.nonzero(cpx)])
    return crop(cpx[extent], cpx.shape)


def blockwise(img:np.ndarray, bs:int):
    """ yield blocks of size bs for a 3D array """
    h,w,l = img.shape
    b1,b2,b3 = bs
    for i in np.r_[:h:b1]:
        for j in np.r_[:w:b2]:
            for k in np.r_[:l:b3]:
                yield np.s_[i:i+b1,j:j+b2,k:k+b3]

def dct_scale(img:np.ndarray, N:int, M:int, bs:int=8)->np.ndarray:
    """ block-wise dct, scale image by factor of N/M
    # https://doi.org/10.1109/ICALIP.2008.4590237
    # https://doi.org/10.1109/ICIP.2004.1421685
    """
    d = img.ndim
    bs = np.tile(bs,d)[:d] # block size
    N = np.tile(N,d)[:d]
    M = np.tile(M,d)[:d]
    print('scale by ',N/M)
    # ------------------------
    from scipy.fftpack import dctn,idctn
    def scale(im,n,m):
        # scale by n/m (upsample by zero-pad, downsample by cropping)
        l = np.min(np.c_[n,m],axis=1)
        s = np.s_[:l[0],:l[1],:l[2]]
        shp = np.array(im.shape,dtype=float)
        nb = np.ceil(shp/m)
        cimg = crop(im,nb*m)
        out = np.zeros(tuple((n*nb).astype(int)))
        for i,j in zip(blockwise(out,n),blockwise(cimg,m)):
            out[i][s] = dctn(cimg[j],norm='ortho')[s]
            out[i] = idctn(out[i],norm='ortho')
        return crop(out,np.ceil(shp*n/m))
    # ------------------------
    # img scaled by N/M (scale by N/bs then bs/M)
    outshp = np.floor(np.array(img.shape)*N/M)
    return crop(scale(scale(img,N,bs),bs,M),outshp)


# ------------------------------------------------------
def threshold_by_edge(fp:np.ndarray)->np.ndarray:

    # threshold by left edge value
    mask = np.ones_like(fp,dtype=bool)
    mask[tuple([slice(1,None)]*fp.ndim)] = 0
    cut = np.max(fp[mask])

    binary = np.zeros_like(fp)
    binary[(np.abs(fp)>zero) & (fp>cut)] = 1
    if debug: save_3d_tiff("debug_3_threshold.tif",binary)
    return binary


def select_central_object(fp:np.ndarray)->np.ndarray:

    import scipy.ndimage as ndimage

    binary = np.abs(fp)
    binary[binary>zero] = 1
    binary[binary<=zero] = 0

    # cluster by connectivity
    struct = ndimage.morphology.generate_binary_structure(fp.ndim,1).astype("uint8")
    label, nlabel = ndimage.label(binary,structure=struct)

#    # select cluster in the center
#    select = label[tuple((np.array(binary.shape)*0.5).astype(int))]

    # select largest cluster
    select = np.argmax(np.bincount(np.ravel(label))[1:])+1

    binary[label!=select] = 0

    fp[binary==0] = 0
    if debug: save_3d_tiff("debug_4_extracted.tif",fp)
    return fp


def get_central_object_extent(fp:np.ndarray)->list:

    fp_cut = threshold_by_edge(np.abs(fp))
    need = select_central_object(fp_cut)

    # get extend of cluster
    extent = [np.max(s)+1-np.min(s) for s in np.nonzero(need)]
    return extent


def get_cc_oversample_ratio(fp:np.ndarray)->np.ndarray:

    # autocorrelation
    if debug: save_3d_tiff("debug_1_input.tif",np.abs(fp))
    acp = np.fft.fftshift(np.fft.ifftn(np.abs(fp)**2.))
    aacp = np.abs(acp)
    if debug: save_3d_tiff("debug_2_autocorr.tif",aacp)

    # get extent
    blob = get_central_object_extent(aacp)

    # correct for underestimation due to thresholding
    correction = [0.025,0.025,0.0729][:fp.ndim]
    extent = [min(m,s+int(round(f*aacp.shape[i],1)))
              for i,(s,f,m) in enumerate(zip(blob,correction,aacp.shape))]

    # oversample ratio
    oversample = [2.*s/(e+(1-s%2)) for s,e in zip(aacp.shape,extent)]

    return np.round(oversample,3)


def real_match_oversample(
    img: np.ndarray,
    fr: Union[list,np.ndarray,None] = None,
    to: Union[list,np.ndarray,None] = None,
):
    """ crop/pad real space object to match oversample ratios """

    # adjustment needed to match oversample ratio
    change = img.shape*(to/fr-1.)

    # crop
    crop = np.copy(change)
    crop[crop>0] = 0
    lo = np.round(-0.5*crop,0).astype(int)
    hi = -(lo + np.mod(crop,2).astype(int))
    hi[hi==0] = np.array(img.shape)[hi==0]
    img = img[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]

    # zero pad
    pad = np.copy(change)
    pad[pad<0] = 0
    lo = np.round(0.5*pad,0).astype(int)
    hi = lo + np.mod(np.round(pad,0),2).astype(int)
    pads = tuple([(l,h) for l,h in zip(lo,hi)])
    img = np.pad(img,pads,'constant',constant_values=0)

    return img


# ------------------------------------------------------
def img_variance(img, ksize=(3,3,3), periodic=False):

    ksize = np.tile(ksize,img.ndim)[:img.ndim]
    padsize = tuple([(k-1,k-1) for k in ksize])
    padmode = 'wrap' if periodic else 'reflect'
    pad_data = np.pad(img,padsize,padmode)

    from scipy.ndimage import uniform_filter
    variance = (uniform_filter(pad_data**2.,ksize)-
                uniform_filter(pad_data,ksize)**2.)
    variance[variance<zero] = 0
    variance -= np.min(variance)
    if np.any(variance>zero):
        variance /= np.max(variance)
        variance *= 255
    s_ = tuple([slice(r-1,1-r) for r in ksize]) # unpad
    return variance[s_]


def trim_zeros(arr):
    """
    Index to trim the leading and trailing zeros in a N-D array.

    :param arr: numpy array
    :returns: slice object
    """
    s = []
    for dim in range(arr.ndim):
        start, end = 0, -1
        slice_ = [slice(None)]*arr.ndim

        while True:
            slice_[dim] = start
            start += 1
            if start==arr.shape[dim]: break
            if np.any(arr[tuple(slice_)]): break
        start = max(start-1, 0)

        while True:
            slice_[dim] = end
            end -= 1
            if abs(end)==arr.shape[dim]: break
            if np.any(arr[tuple(slice_)]): break
        end = arr.shape[dim] + min(-1, end+1) + 1

        s.append(slice(start,end))
    return tuple(s)


def ix_argmax(img):
    """return array index of max value """
    return np.unravel_index(np.argmax(img,axis=None),img.shape)


def get_oversample_ratio(fp:np.ndarray)->np.ndarray:
    return get_cc_oversample_ratio(fp)

    #afp = np.abs(fp)
    afp = fp
    if debug: save_3d_tiff("debug_1_input.tif",afp)

    # no signal
    if afp.size==1 or not np.any(afp>zero):
        return None

    afp[afp<=zero] = 0
    uniq = np.unique(afp)

    print('afp\n',afp)
    print('uniq',uniq)
    ix_max = ix_argmax(afp)
    print('ix_max',ix_max)

    # one pixel
    if len(uniq)==1:
        return np.array(fp.shape)

    # detect edge of central blob
    vr = img_variance(afp)
    vr_trim = vr[trim_zeros(vr)]
    vr_extent = np.array(vr_trim.shape)
    if debug: save_3d_tiff("debug_1_var.tif",vr)
    print('vr\n',vr)
    print('vr_extent',vr_extent)

    # estimate oversample ratio by extent of blob
    afp_shape = np.array(afp.shape)
    ix = vr_extent<afp_shape
    oversample = afp_shape[:]
    oversample[ix] = vr_extent[ix]-2
    return oversample



#    vr[vr<=vr[ix_max]] = 0

    return 'todo'


# ------------------------------------------------------
def to_vtk(outfile:str, data:dict)->None:

    # conda install -c conda-forge pyevtk

#    from pyevtk.hl import gridToVTK
#    size = [0,0,0]
#    for t in data:
#        data[t] = np.squeeze(data[t])
#        size = [max(size[i],data[t].shape[i]) 
#                for i in range(data[t].ndim)]
#
#    xyz = np.mgrid[0:size[0],0:size[1],0:size[2]]
#
#    values = data
#    for s in values:
#        print("vtk output,",s,values[s].shape,
#              np.min(values[s]),np.max(values[s]))
#
#    gridToVTK(outfile, xyz[0].copy(), xyz[1].copy(), 
#              xyz[2].copy(),pointData=values)

    from pyevtk.hl import imageToVTK
    imageToVTK(outfile, pointData=data)

# ------------------------------------------------------
def gauss(dims, sigs):
    """ dimensionally agnostic gaussian """

    if (len(sigs) != len(dims)):
        return None

    griddims = []
    for d in dims:
        griddims.append( slice(-d/2+1,d/2+1) )
    grid = np.ogrid[ griddims ]

    g = np.ones(dims)
    d = 0
    for sig in sigs:
        if sig == 0:
          g *= (grid[d]==0).astype(float)
        else:
          g *= np.exp(-0.5*grid[d]*grid[d]/sig**2)
        d += 1
    return g


def gauss_conv_fft(arr, sigs):
    """ save an fft by creating guassian in ft space """

    dims = arr.shape
    if len(sigs) != len(dims):
        return None
    tot = np.sum(arr)
    sigk = np.array(dims)/2.0/np.pi/sigs

    # gaussian needs to be wrap around to match the ft of arr
    gk = np.fft.fftshift(gauss(dims, sigk))

    arrk = np.fft.fftn(arr)
    convag = np.fft.ifftn(arrk*gk)
    convag = np.where(convag<0.0, 0.0, convag)
    convag *= tot/np.sum(convag)
    return convag
