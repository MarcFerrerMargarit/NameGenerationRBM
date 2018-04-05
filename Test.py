import argparse
import pickle

from sklearn.model_selection import train_test_split
from GridEncoder import GridEncoder
import Utils
from ShortTextCodec import ShortTextCodec, BinomialShortTextCodec
from RBM import CharBernoulliRBM, CharBernoulliRBMSoftmax
import numpy as np
from sample import  horizontal_cb, print_columns
import Sampling
import time
from timeit import default_timer as timer

if __name__ == '__main__':
    codec_kls = ShortTextCodec
    codec = codec_kls('', 10, 0, True, False)
    # CODEC SPANISH WORDS
    # codec_kls = ShortTextCodec
    # codec = codec_kls('áéíóúñÁÉÍÓÚÑ', 10, 0, True, False)
    codec.debug_description()
    model_kwargs = {'codec': codec,
                    'n_components': 256,
                    'learning_rate': 0.1,
                    'lr_backoff': False,
                    'n_iter': 10,
                    'verbose': 1,
                    'batch_size': 10,
                    'weight_cost': 0.0001,
                    }
    kls = CharBernoulliRBMSoftmax
    rbm = kls(**model_kwargs)
    # FILE TO READ (ENGLISH NAMES OR SPANISH WORDS)
    vecs = Utils.vectors_from_txtfile("./names.txt", codec)
    train, validation = train_test_split(vecs, test_size=0.5)
    rbm.fit(train, validation)
    hidden_test = []
    for i in range(10):
        tmp = []
        for j in range(256):
            tmp.append(0)
        hidden_test.append(tmp)
    hidden_test = np.asarray(hidden_test)
    print(rbm._sample_visibles(hidden_test))
    # sample_indices = [1000 - 1]
    # kwargs = dict(start_temp=1.0, final_temp=1.0, sample_energy=False,
    #               callback=horizontal_cb)
    # vis = Sampling.sample_model(rbm, 30, 1000, sample_indices, **kwargs)
    # print_columns(rbm.codec.maxlen)
    # fe = rbm._free_energy(vis)
    # print('Final energy: {:.2f} (stdev={:.2f})\n'.format(fe.mean(), fe.std()))
