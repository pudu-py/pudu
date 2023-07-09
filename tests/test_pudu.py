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
from pudu import standards
from pudu import masks as msk
from pudu import perturbation as ptn

# import pudu8 as pudu
# import perturbation8 as ptn
# import masks8 as msk
# import standards8 as standards


TESTDATA_FEATURES = os.path.join(os.path.dirname(__file__), 'data/features.txt')
TESTDATA_TARGETS = os.path.join(os.path.dirname(__file__), 'data/targets.txt')
TESTDATA_LDA = os.path.join(os.path.dirname(__file__), 'data/lda_model.sav')
TESTDATA_PCA = os.path.join(os.path.dirname(__file__), 'data/pca_model.sav')
TESTDATA_RESULTS = os.path.join(os.path.dirname(__file__), 'data/pudu_test_results8.txt')
MNIST_MODEL = os.path.join(os.path.dirname(__file__), 'data/mnist_class.h5')
MNIST_RESULTS = os.path.join(os.path.dirname(__file__), 'data/act_results_mnist8.txt')
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

        # just vanilla settings except for window=200
        imp.importance(window=200)
        imp.speed(window=200)
        imp.synergy(window=200)
        
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
        sh = np.array(x).shape
        scope, window, padd, evolution, total = standards.params_std(None, sh, None, 1, 'center', 0)

        ptn_results = spep.load(TEST_PERTURBATIONS)
        vector = [i*0.01 for i in range(sh[2])]
        vector = np.array(vector)[np.newaxis, np.newaxis, :, np.newaxis]

        f = [ptn.Negative(), ptn.Log(), ptn.Exp(), ptn.Binary(), ptn.Sinusoidal(), ptn.Gaussian(), ptn.Tanh(),
                ptn.Sigmoid(), ptn.ReLU(), ptn.LeakyReLU(), ptn.Softplus(), ptn.Inverse(), ptn.Offset(), 
                ptn.Constant(), ptn.Custom(custom=vector), ptn.UpperThreshold(), ptn.LowerThreshold()]

        for i,j in enumerate(f):
            a = j.apply(x, 0, 0, window, 0)
            a = np.array(a[0]).reshape((sh[2]))
            b = ptn_results[i]
            for k, l in zip(a, b):
                self.assertAlmostEqual(k, l, places=tp)

        ### masks
        masks_test = spep.load(TEST_MASKS)
        sh = np.array(x).shape
        scope, window, padding, evolution, total  = standards.params_std(0, sh, None, 1, 'center', 0)
        padding = standards.calc_pad(padding, scope, window)

        f = [msk.Percentage(), msk.Quantity(qty=50), msk.EveryOther(), msk.Pairs(), msk.Odds(), msk.All()]
        for i,j in enumerate(f):
            mask_temp = []
            for k in range(100):
                a = j.apply(k, total)
                mask_temp.append(a)
            for k, l in zip(mask_temp, masks_test[i]):
                self.assertAlmostEqual(k, l, places=tp)


        ### 2d activations
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
        imp.reactivations(layer=2, slope=0, p=0.005, window=(5, 5), perturbation=ptn.Positive(delta=0.1))

        for a, b in zip(imp.fac, fac):
            self.assertAlmostEqual(a, b, places=tp)
        
        for a, b in zip(imp.uac[338:364], uac):
            self.assertAlmostEqual(a, b, places=tp)


