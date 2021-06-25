import numpy as np


# load model setup
import json
with open('nn_model/setup.json') as f:
    setup = json.load(f)
xs, ys = setup['xs'], setup['ys']
dimension = setup['dimension']
data_format = setup['data_format']

# import nn model
from nn_model import model as m
model = m.load_model(
    jfile='nn_model/model.json',
    wfile='nn_model/weights.1574198591.hdf5'
)

# setup tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
print(f'tensorflow version = {tf.__version__}')


def nn_predict():

    data = {}

    # load and format input
    fp = np.load("ft_downsampled_32.npy")
    fp = np.expand_dims(fp,axis=0)
    fp = np.expand_dims(fp,axis=-1)
    if data_format == 'channels_first':
        swap = 'aijkl->alijk' if dimension=='3D' else 'aijk->akij'
        fp = np.einsum(swap,fp)  # swap axis
    fp = fp.astype(np.float32)
    data[xs[0]] = fp


    # predict
    predict = model.predict(fp)
    for k,s in enumerate(ys):
        data['predict_%s'%s] = predict[k]

    shape32 = data['predict_real_intensity']
    phase32 = (data['predict_real_phase'] - 0.5)*2.*np.pi

    nn = np.squeeze(shape32) * np.exp(1j*np.squeeze(phase32))
    np.save('nn_output_32.npy', nn)


if __name__ == '__main__':
    nn_predict()
