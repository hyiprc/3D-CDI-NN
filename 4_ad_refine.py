import numpy as np
from skimage import morphology
import hy

# setup tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
print(f'tensorflow version = {tf.__version__}')
tf_device = '/cpu:0'


setup = {

# based on 3_compare_re_nn.py
'in_size': 64,
'undersample_target': False,

# Number of cycles and AD iterations
'n_cycles': 1,
'n_iters': 1000,
'learn_rate': 1e-2,

# Shrink wrap variables (Gaussian smoothing)
'dilate_support': 0, # NN as initial shrink wrap support
'isig': 1, # Sigma of G for first support update
'fsig': 1, # Sigma for final support update
'severy': 50, # Update support every how many iterations?
'sfrac': 0.15, # Fraction of maximum intensity to use for shrink wrap boundary
}

if setup['undersample_target']:
    typ = 'downsampled'
else:
    typ = 'crop'

setup['target'] = f"ft_{typ}_{setup['in_size']}.npy" 


# Shrink wrap support with increasing iterations
# by scaling guassian smoothening
def update_support(obj:np.ndarray, i:int, setup:dict):
    # Only shrink support during the first cycle
    ntot = setup['n_iters']/setup['n_cycles']
    # Calculate initial and final update points
    xo, xf = 1., ntot/setup['severy'] 
    yo, yf = setup['isig'], float(setup['fsig'])
    i /= float(setup['severy']) #Count in number of updates

    if xf == 1:
        # 0 division of only 1 update done
        sig = (setup['isig']+setup['fsig'])/2.
    else:
        # Linearly scale sigma between initial and final values
        sig = yo+(i-xo)*((yf-yo)/(xf-xo))
    #if not i%setup['severy']: print ("%d Real space sigma"%i, sig)
    rimage = np.abs(obj)

    smooth = np.abs(hy.gauss_conv_fft(rimage,[sig,sig,sig]))
    smooth /= smooth.max()
    supp = (smooth>=setup['sfrac'])*1 # Threshold as fraction of max
    #xyz_save(supp,'visuals/supp%d.xyz' %i)
    return supp


# Simple Reconstruction: 
tf.compat.v1.disable_eager_execution()
def reconstruct(
    target_diffraction: np.ndarray,
    initial_guess: np.ndarray,
    initial_support: np.ndarray,
    setup: dict,
):

    guess0 = np.copy(initial_guess)
    support0 = np.copy(initial_support)

    tf.compat.v1.reset_default_graph()
    with tf.device(tf_device):

        tf_diffs = tf.constant(target_diffraction, dtype='float32')

        tf_obj_real = tf.Variable(np.real(guess0), dtype='float32')
        tf_obj_imag = tf.Variable(np.imag(guess0), dtype='float32')
        tf_obj = tf.complex(tf_obj_real, tf_obj_imag)

        tf_support = tf.compat.v1.placeholder(tf.float32, shape=support0.shape)
        tf_support = tf.complex(tf_support, tf.zeros_like(tf_support))
        tf_obj *= tf_support

        # Finally the loss function
        exitwave = tf.abs(tf.signal.fft3d(tf_obj))
        exitwave /= tf.reduce_max(exitwave)
        loss = tf.reduce_sum((exitwave - tf_diffs)**2)

        print('learning rate: %g\n'%setup['learn_rate'])
        opt = tf.compat.v1.train.AdamOptimizer(setup['learn_rate'])
        minimize_op = opt.minimize(loss)

        sess_config = tf.compat.v1.ConfigProto()
        #sess_config.gpu_options.allow_growth = True 
        #sess_config.allow_soft_placement = True
        session = tf.compat.v1.Session(config=sess_config)
        session.run(tf.compat.v1.global_variables_initializer())

        lossvals = []
        lowest = np.inf

        print('update support every: %d'%setup['severy'])
        print('sigma of G: from %g to %g'%(setup['isig'],setup['fsig']))
        print('shrink wrap boundary threshold fraction: %g'%setup['sfrac'])

    print('\nrunning %d iterations ...'%setup['n_iters'])
    for i in range(setup['n_iters']):

        lossval, _ = session.run([loss, minimize_op], 
                                 feed_dict={tf_support: support0})
        lossvals.append(lossval)

        if i % 100 == 0:
            print(f"{i}| current loss {lossval:4.3g}, "+
                  f"loss before last shrinkwrap {lowest:4.3g}")
        if (lowest - lossval) < 0.1 * lowest or i % setup['severy'] != 0:
            continue

        lowest = lossval
        recon = session.run(tf_obj, feed_dict={tf_support:support0})
        if i % setup['severy'] == 0:
            support0 = update_support(recon, i, setup)

    print('lowest lossval: ',np.min(lossvals))

    support0 = update_support(recon, i, setup)

    recon = recon * support0
    recon /= np.max(np.abs(recon))
    return {
        'output': recon,
        'loss': lossvals,
        'support': support0,
    }


def run_ad(setup):

    # load the diffraction patterns 
    fp = np.fft.fftshift(np.load(setup['target']))
    print("ft input shape =",fp.shape,", minmax =",np.min(fp),np.max(fp))

    # initial guess
    guess = np.load(f"ad_input_{setup['in_size']}.npy")
    support = (np.abs(guess) > 1e-4)
    support = morphology.dilation(support, selem=morphology.ball(setup['dilate_support']))

    # reconstruct
    refined = reconstruct(fp, guess, support, setup)

    # output
    hy.to_vtk("compare/ad_output", {
        'shape': np.abs(refined['output']),
        'phase': np.angle(refined['output']),
    })

    np.save(f"ad_output_{setup['in_size']}.npy",refined['output'])


if __name__ == '__main__':
    run_ad(setup)
