
#import keras.models as krm
#import keras.layers as krl
#import keras.utils as kru

import tensorflow as tf
krm = tf.keras.models
krl = tf.keras.layers
kru = tf.keras.utils

# -----------------------------

def autoencoder(n=1,dim='2D',shp=(32,32,1),dropout=0.0,
                data_format='channels_last',act='relu',**kwargs):

    if dim == '2D':
        def MaxPool():
            return krl.MaxPooling2D((2,2),padding='same',data_format=data_format)
        def UpSampling():
            return krl.UpSampling2D((2,2), data_format=data_format)
        conv_size = (3,3)

    elif dim == '3D':
        def MaxPool():
            return krl.MaxPooling3D((2,2,2),padding='same',data_format=data_format)
        def UpSampling():
            return krl.UpSampling3D((2,2,2), data_format=data_format)
        conv_size = (3,3,3)

    def Conv(filters=16, activation=act, 
             kernel_size=conv_size, input_shape=None):
        setup = {
            'filters':filters,
            'activation':activation,
            'kernel_size':kernel_size,
            'padding':'same',
            'data_format':data_format,
        }
        if input_shape is None:
            if dim=='2D': return krl.Conv2D(**setup)
            return krl.Conv3D(**setup)
        if dim=='2D': return krl.Conv2D(input_shape=input_shape,**setup)
        return krl.Conv3D(input_shape=input_shape,**setup)

    def Drop(x):
        if dropout==0.0: return x
        else: return krl.Dropout(dropout)(x)

    input_img = tf.keras.Input(shape=shp)
    # encoder
    x = Drop(input_img)
    for res in [32,64,128]:
        x = Drop(Conv(res)(x)) ;  x = Drop(Conv(res)(x)) ; x = MaxPool()(x)
    encoder = x
    # decoder
    decoders = []
    for _ in range(n):
        x = encoder
        for res in [128,64,32]:
            x = Drop(Conv(res)(x)); x = Drop(Conv(res)(x)); x = UpSampling()(x)
        decoders.append(Conv(1,act)(x))  # sigmoid
    if n==1: decoders = decoders[0]
    return krm.Model(inputs=input_img, outputs=decoders)

def autoencoder2(n=2,dim='2D',shps=[(32,32,1),(32,32,1)],dropout=0.0,
                data_format='channels_last',**kwargs):

    if dim == '2D':
        def MaxPool():
            return krl.MaxPooling2D((2,2),padding='same',data_format=data_format)
        def UpSampling():
            return krl.UpSampling2D((2,2), data_format=data_format)
        conv_size = (3,3)

    elif dim == '3D':
        def MaxPool():
            return krl.MaxPooling3D((2,2,2),padding='same',data_format=data_format)
        def UpSampling():
            return krl.UpSampling3D((2,2,2), data_format=data_format)
        conv_size = (3,3,3)

    def Conv(filters=16, activation=act, 
             kernel_size=conv_size, input_shape=None):
        setup = {
            'filters':filters,
            'activation':activation,
            'kernel_size':kernel_size,
            'padding':'same',
            'data_format':data_format,
        }
        if input_shape is None:
            if dim=='2D': return krl.Conv2D(**setup)
            return krl.Conv3D(**setup)
        if dim=='2D': return krl.Conv2D(input_shape=input_shape,**setup)
        return krl.Conv3D(input_shape=input_shape,**setup)

    def Drop(x):
        if dropout==0.0: return x
        else: return krl.Dropout(dropout)(x)

    inputs = []
    for shp in shps:
        inputs.append(krm.Input(shape=shp))
    # decoder
    decoders = []
    for i in range(n):
        x = Drop(inputs[i])
        for res in [32,64,128]:
            x = Drop(Conv(res)(x)) ;  x = Drop(Conv(res)(x)) ; x = MaxPool()(x)
        for res in [128,64,32]:
            x = Drop(Conv(res)(x)); x = Drop(Conv(res)(x)); x = UpSampling()(x)
        decoders.append(Conv(1,act)(x))  # sigmoid
    if n==1: decoders = decoders[0]
    return krm.Model(inputs=inputs, outputs=decoders)

def save_model(model,jfile='tmp/model.json',wfile='tmp/weights.h5'):
    with open(jfile, 'w') as f:
        f.write(model.to_json())
    if not wfile is None:
        model.save_weights(wfile)

def load_model(jfile='tmp/model.json',wfile='tmp/weights.h5'):
    with open(jfile, 'r') as f:
        model = krm.model_from_json(f.read())
        model.load_weights(wfile)
    return model

# -----------------------------

class ModelMGPU(krm.Model):
    def __init__(self, ser_model, gpus):
        pmodel = kru.multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
    	    return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

