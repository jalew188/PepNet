import numpy as np
import typing
import tensorflow.keras as k
from tensorflow.keras import backend as K
from keras import Input
from tensorflow.keras.layers import (
    Conv1D, Add, UpSampling1D, AveragePooling1D,
    Activation, ZeroPadding1D
)
from tensorflow_addons.layers import InstanceNormalization

Alist = list('ACDEFGHIKLMNPQRSTVWYZ')
clist = ['*'] + Alist + [']', '[']
out_dim = len(clist)
max_peplen = 30
outlen = max_peplen + 2
max_mz = 2048.0
mz_bins = int(max_mz/0.1)
sp_dim = 4
info_dim = 2
max_charge = 8

def build_model(
    sp_dim=sp_dim, mz_bins=mz_bins, 
    info_dim=info_dim, max_charge=max_charge, 
)->typing.Tuple[k.Model, k.Model, k.Model]:
    inp = Input(shape=[sp_dim,mz_bins], name='sp_inp')
    mz_inp = Input(shape=[info_dim], name='mz_inp')
    c_inp = Input(shape=[max_charge], name='charge_inp')
    model_inputs = [inp, mz_inp, c_inp]

    spmodel = sp_net(
        sp_dim=sp_dim,
        mz_bins=mz_bins,
        info_dim=info_dim,
        max_charge=max_charge
    )
    sp_vector = spmodel(model_inputs)

    aux_outputs = auxiliary_tasks(sp_vector)
    
    final_pep = Conv1D(out_dim, kernel_size=1, padding='same', use_bias=1)(sp_vector)
    final_pep = Activation('softmax', name='peps', dtype='float32')(final_pep)

    full_model = k.models.Model(inputs=model_inputs, outputs=aux_outputs + [final_pep], name='full_model')
    novo = k.models.Model(inputs=model_inputs, outputs=final_pep, name='denovo')

    return full_model, novo, spmodel

def auxiliary_tasks(v1, act='relu'):        
    def vec_dense(x, nodes, name, act='sigmoid', layers=tuple()):
        for l in layers: x = res(x, l, 3, act='relu')
        x = k.layers.GlobalAveragePooling1D()(x)
    #     x = k.layers.Flatten()(x)
        x = k.layers.Dense(nodes, activation=act, name=name, dtype='float32')(x)
        return x

    aux_outputs = []

    aux_outputs.append(vec_dense(v1, 1, name='mass'))
    aux_outputs.append(vec_dense(v1, outlen, act='softmax', name='length'))
    aux_outputs.append(vec_dense(v1, 1, name='rk'))
    aux_outputs.append(vec_dense(v1, max_charge, act='softmax', name='charge'))

    #aux exist:
    x = v1
    x = k.layers.GlobalAveragePooling1D()(x)
    x = k.layers.Dense(len(Alist))(x)
    x = Activation('sigmoid', name='exist', dtype='float32')(x)
    aux_outputs.append(x)

    #aux compose:
    x = v1
    x = k.layers.Permute((2, 1))(x)
    x = res(x, len(Alist), 1, act=act)
    x = k.layers.Permute((2, 1))(x)

    x = k.layers.Conv1D(max_peplen, kernel_size=1, padding='same')(x)
    x = k.layers.Activation('softmax', name='nums', dtype='float32')(x)
    aux_outputs.append(x)

    #aux AA pairs:
    x = v1
    x = k.layers.GlobalAveragePooling1D()(x)
    x = k.layers.Dense(400)(x)
    x = k.layers.Activation('sigmoid', name='di', dtype='float32')(x)
    aux_outputs.append(x) # don't merge

    return aux_outputs

def sp_net(
    sp_dim=sp_dim, mz_bins=mz_bins, 
    info_dim=info_dim, max_charge=max_charge, 
    act='relu',
):
    inp = Input(shape=[sp_dim,mz_bins], name='sub_sp_inp')
    mz_inp = Input(shape=[info_dim], name='sub_mz_inp')
    c_inp = Input(shape=[max_charge], name='sub_charge_inp')

    v1 = k.layers.Permute((2, 1))(inp)

    def sub_net(v1, act='relu'):
        for i, l in enumerate([8, 12]):
            v1 = res(v1, l, 7, add=1, act=act, strides=2)

        lst = []

        fils = np.int32([16, 24, 32, 48, 64]) * 12
        tcn = np.int32([8, 7, 6, 5, 4, ]) + 1

        for i, (l, r) in enumerate(zip(fils, tcn)):
            if i > 0:
                v1 = res(v1, l, 9, add=1, act=act, strides=2)

            ext = r - 5
            if r > ext:
                r = r - ext
                ks = int(5 * 2 ** ext) - 1
            else:
                ks = int(5 * 2 ** int(r - 1)) - 1
                r = 1

            v1 = res(v1, l, ks, tcn=r, add=1, act=act)

            lst.append(v1)
        return v1, lst

    v1, lst = sub_net(v1)
    v1 = bottomup(lst[2:])

    v1 = k.layers.Permute((2, 1))(v1)
    v1 = res(v1, outlen, 1, act=act, add=0)
    v1 = k.layers.Permute((2, 1))(v1)

    l_size = K.int_shape(v1)[-2]
    infos = k.layers.Concatenate(axis=-1)([mz_inp, c_inp]) # meta infos
    infos = k.layers.Reshape((l_size, 1))(k.layers.Dense(l_size, activation='sigmoid')(infos))
    v1 = k.layers.Concatenate(axis=-1)([v1, infos])

    return k.models.Model([inp, mz_inp, c_inp, ], v1, name='sp_net')

def merge(
    o1, c1, strides=1, mact=None
):
    layers = K.int_shape(c1)[-1]

    if strides > 1 or K.int_shape(o1)[-1] != layers:
        if strides > 1:
            o1 = ZeroPadding1D((0, strides-1))(o1)
            o1 = AveragePooling1D(strides)(o1)

        if K.int_shape(o1)[-1] != layers:
            o1 = Conv1D(layers, kernel_size=1, padding='same')(o1)

        o1 = InstanceNormalization()(o1) # no gamma zero, main path

    if mact is None:
        return Add()([o1, c1])
    else:
        return Activation(mact)(Add()([o1, c1]))

def conv(
    x, layers, kernel, act='relu', dilation_rate=1,
    tcn=1, strides=1
):
    if isinstance(kernel, int): kernel = (kernel,)
    for i, ks in enumerate(kernel):
        if i > 0: x = Activation(act)(x)

        model = Conv1D(
            layers, kernel_size=ks, padding='same',
            strides=strides, dilation_rate=dilation_rate
        )
        x = model(x)
        x = InstanceNormalization()(x)

        for r in range(1, tcn):
            assert strides == 1 and dilation_rate == 1

            x = Activation(act)(x)
            model = Conv1D(
                layers, kernel_size=kernel,
                padding='same', dilation_rate=2**r,
            )
            x = model(x)
            x = InstanceNormalization()(x)
    return x

def res(
    x, l, ks, add=1, act='relu',
    strides=1, tcn=1,
):
    xc = conv(x, l, ks, act=act, strides=strides, tcn=tcn)
    if add: xc = merge(x, xc, mact=None, strides=strides)
    x = Activation(act)(xc) #final activation, xc to x naming
    # if pooling == 2 or pooling == 'ave':
    #     x = AveragePooling1D(pool)(x)
    return x

def bottomup(fu, act='relu'):
    v1 = fu[0]
    fu = fu[1:] # first is v1
    for u in fu:
        v1 = res(
            v1, K.int_shape(u)[-1], 5, act=act, 
            strides=2, add=0
        )
        v1 = k.layers.Add()([v1, u])
#         v1 = Activation(act)(v1)
    
    return v1
