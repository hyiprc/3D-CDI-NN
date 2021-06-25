import numpy as np
import hycpy as hy
import hy
fft = hy.fft


out_size = 64
undersample_target = False  # make target undersample (match with NN input)?


# ----- Resize reconstructed -----

# load and get clean object and center it
target = np.load('reconstruct/reconstructed.npy')
l = target.shape
target = hy.select_central_object(target)
ix = hy.trim_zeros(target)
target = hy.crop(target[ix],l)
#oversample0 = hy.get_oversample_ratio(fft(target))
#print('RE real',target.shape,', oversample:',oversample0)

# get target oversample ratio
fp = hy.load_3d_tiff('original.tif')
original_oversample = hy.get_oversample_ratio(fp)
print('FT original',fp.shape,', oversample:',original_oversample)
if undersample_target:
    target_oversample = np.array(open(f'ft_downsampled_{out_size}.txt').read().split()[1:],dtype=float)
    target = hy.real_match_oversample(target, fr = original_oversample, to = target_oversample)
    print('RE real',target.shape,', oversample:',target_oversample)
else:
    target_oversample = original_oversample

# resize to final output size
#target_scaled = hy.cpx_real_resize(target,out_size)
#target_scaled_oversample = hy.get_oversample_ratio(fft(target_scaled))
#print('RE real',target_scaled.shape,', oversample:',target_scaled_oversample)

ft_target = hy.fft(target)
ft_target = hy.crop(ft_target,64)
target_scaled = hy.ifft(ft_target)


# normalize
norm = np.max(np.abs(target_scaled))
target_scaled /= norm
target_scaled[np.abs(target_scaled)<1e-2] = 0.0
target_scaled = hy.phase_wrap(target_scaled,threshold=0.3)

hy.to_vtk("compare/recon_output", {
    'shape': np.abs(target_scaled),
    'phase': np.angle(target_scaled)
})

#np.save(f'reconstructed_{out_size}.npy',target_scaled)


# ----- Resize NN prediction -----
print('')

# load NN predicted
nn = np.squeeze(np.load('nn_output_32.npy'))
#print('NN real',nn.shape,', est. oversample:',hy.get_oversample_ratio(fft(nn)))

# scale to target shape (more pixel to work with)
predicted = hy.cpx_real_resize(nn, target.shape[0])
predicted = hy.select_central_object(predicted)
oversample1 = np.array(open('ft_downsampled_32.txt').read().split()[1:],dtype=float)
#oversample1 = hy.get_oversample_ratio(fft(predicted))-0.4
print('NN real',predicted.shape,', oversample:',oversample1,' (from file)')

# adjust predicted to match target oversample
predicted = hy.real_match_oversample(predicted, fr = oversample1, to = target_oversample)
oversample2 = hy.get_oversample_ratio(fft(predicted))
print('NN real',predicted.shape,', oversample:',oversample2)

# resize to final output size
predicted_scaled = hy.cpx_real_resize(predicted,out_size)
#oversample3 = hy.get_oversample_ratio(fft(predicted_scaled))
#print('NN real',predicted_scaled.shape,', oversample:',oversample3)

# normalize
norm = np.max(np.abs(predicted_scaled))
predicted_scaled /= norm
predicted_scaled[np.abs(predicted_scaled)<1e-2] = 0.0


hy.to_vtk("compare/nn_output", {
    'shape': np.abs(predicted_scaled),
    'phase': np.angle(predicted_scaled)
})

np.save(f'ad_input_{out_size}.npy',predicted_scaled)
