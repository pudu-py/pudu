#!/usr/bin/env python

"""Tests for `pudu` package."""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tensorflow import keras
from keras.models import load_model
import spectrapepper as spep
import numpy as np
import unittest
import pickle
import os

from pudu import pudu, standards, masks, perturbation
# import pudu7 as pudu
# import standards, error_handler, masks, perturbation


TESTDATA_FEATURES = os.path.join(os.path.dirname(__file__), 'data/features.txt')
TESTDATA_TARGETS = os.path.join(os.path.dirname(__file__), 'data/targets.txt')
TESTDATA_LDA = os.path.join(os.path.dirname(__file__), 'data/lda_model.sav')
TESTDATA_PCA = os.path.join(os.path.dirname(__file__), 'data/pca_model.sav')
TESTDATA_RESULTS = os.path.join(os.path.dirname(__file__), 'data/pudu_test_results.txt')
MNIST_MODEL = os.path.join(os.path.dirname(__file__), 'data/mnist_class.h5')
MNIST_RESULTS = os.path.join(os.path.dirname(__file__), 'data/act_results_mnist.txt')
TEST_PERTURBATIONS = os.path.join(os.path.dirname(__file__), 'data/perturbations.txt')
TEST_MASKS = os.path.join(os.path.dirname(__file__), 'data/masks.txt')

class TestPudu(unittest.TestCase):
    """Tests for `pudu` package."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test(self):
        x = spep.load(TESTDATA_FEATURES)[0]
        x = x[np.newaxis, np.newaxis, :, np.newaxis]
        y = spep.load(TESTDATA_TARGETS)[0][2]

        lda = pickle.load(open(TESTDATA_LDA, 'rb'))
        pca = pickle.load(open(TESTDATA_PCA, 'rb'))

        results = spep.load(TESTDATA_RESULTS)

        def pf(X):
            X = X[0,:,:,0]
            return lda.predict_proba(pca.transform(X))[0]

        imp = pudu.pudu(x, y, pf)

        imp.importance(delta=1, window=200)
        imp.speed(delta=1, window=200)
        imp.synergy(delta=1, inspect=3, window=200)
        
        tp = 6

        for a, b in zip(imp.imp[0,0,:,0], results[0]):
            self.assertAlmostEqual(a, b, places=tp)

        for a, b in zip(imp.imp_rel[0,0,:,0], results[1]):
            self.assertAlmostEqual(a, b, places=tp)

        for a, b in zip(imp.spe[0,0,:,0], results[2]):
            self.assertAlmostEqual(a, b, places=tp)
        
        for a, b in zip(imp.spe_rel[0,0,:,0], results[3]):
            self.assertAlmostEqual(a, b, places=tp)

        for a, b in zip(imp.syn[0,0,:,0], results[4]):
            self.assertAlmostEqual(a, b, places=tp)
        
        for a, b in zip(imp.syn_rel[0,0,:,0], results[5]):
            self.assertAlmostEqual(a, b, places=tp)

        ### perturbations
        perturbations = spep.load('data/perturbations.txt')
        vector = [i*0.01 for i in range(1536)]
        vector = np.array(vector)[np.newaxis, np.newaxis, :, np.newaxis]

        f = ['negative', 'bias', 'log', 'exp', 'binary', 'sinusoidal', 'gaussian', 'tanh',
            'sigmoid', 'relu', 'leakyrelu', 'softplus', 'inverse',
            'offset', 'constant', 'custom', 'upper-threshold', 'lower-threshold']

        for i,j in enumerate(f):
            print(j)
            a = perturbation.function(x, 0, 100, mode=j, custom=vector)
            a = np.array(a[0]).reshape((1536))
            b = perturbations[i]
            for k, l in zip(a, b):
                self.assertAlmostEqual(k, l, places=tp)

        ### masks
        masks_test = spep.load('data/masks.txt')
        sh = np.array(x).shape
        scope, window, padding, _ = standards.params_std(0, sh, None, 1, 'center', 0)
        padding = standards.calc_pad(padding, scope, window)

        f = ['percentage', 'quantity', 'everyother', 'pairs', 'odds', 'all']
        m = []
        for i,j in enumerate(f):
            print(j)
            a = masks.function(sh=sh, padd=padding, scope=scope, window=window, mask_type=j)
            a = np.array(a).reshape((1536))
            b = masks_test[i]
            for k, l in zip(a, b):
                self.assertAlmostEqual(k, l, places=tp)


        ### 2d
        (x, y), (_, _) = keras.datasets.mnist.load_data()
        x = x.astype("float32") / 255
        x = np.expand_dims(x, -1)
        y = y[10]
        x = x[10]
        x = np.expand_dims(x, 0)
        model = load_model(MNIST_MODEL)
        fac, uac = spep.load(MNIST_RESULTS)

        def cnn2d_prob(X):
            X = X[0,:,:,:]
            return MNIST_MODEL.predict(np.array([X, X]), verbose=0)[0] # verbose 0 is important!
        
        imp = pudu.pudu(x, y, cnn2d_prob, model)
        imp.activations(layer=2, slope=0, p=0.005, window=(5, 5), delta=0.1, mode='positive')

        for a, b in zip(imp.fac, fac):
            self.assertAlmostEqual(a, b, places=tp)
        
        for a, b in zip(imp.uac[338:364], uac):
            self.assertAlmostEqual(a, b, places=tp)


