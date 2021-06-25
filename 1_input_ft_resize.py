import numpy as np
import hy


infile = 'original.tif'


for out_size,target_oversample in [
    (32, 1.5),  # undersampled input for NN
    (64, None), # oversampled input for AD
#    (64, 1.5),  # undersampled input for AD
]:

    print(f'\n----- size: {out_size}, oversample: {target_oversample} -----')
    fp = hy.load_3d_tiff(infile)
    print(infile)
    print('input: ',fp.dtype,np.min(fp),np.max(fp),fp.shape)

    oversample = hy.get_oversample_ratio(fp)
    print('FT',fp.shape,', oversample:',oversample)

    # crop
    if target_oversample is None:
        l_crop = out_size
    else:
        l_crop = max(
            int(out_size*min(oversample)/target_oversample),
            out_size
        )
    #print("crop to:",l_crop)
    img = hy.crop(fp,l_crop)
    crop_os = hy.get_oversample_ratio(img)
    print('FT_crop',img.shape,', oversample:',crop_os)

    if l_crop == out_size:
        typ = 'crop'
    else:
        typ = 'downsampled'
        # scale
        block_size = int(2.*(out_size//8))
        print('resize to:',out_size,', block size:',block_size)
        img = hy.dct_scale(img,out_size,img.shape[0],bs=block_size)

    # verify oversample ratio
    predict_os = hy.get_oversample_ratio(img)
    print('\npredict',predict_os)
    target_os = crop_os*out_size/l_crop
    print('target',target_os)
    open(f'ft_{typ}_{out_size}.txt','w').write(
        'oversample_ratio '+' '.join(['%.3f'%s for s in target_os]))

    # output
    outfile = f'ft_{typ}_{out_size}.tif'
    hy.save_3d_tiff(outfile,img)

    # for neural network
    img[img<0] = 0.0
    img = img**0.5  # intensity to amplitude
    img /= np.max(img)
    np.save(f'ft_{typ}_{out_size}.npy',img.astype(np.float32))
    print('output: ',img.dtype,np.min(img),np.max(img),img.shape)
