#!/usr/bin/env python

"""Tests for `pudu` package."""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import spectrapepper as spep
from pudu import pudu
# import pudu as pudu
import numpy as np
import unittest
import pickle
import os

TESTDATA_FEATURES = os.path.join(os.path.dirname(__file__), 'features.txt')
TESTDATA_TARGETS = os.path.join(os.path.dirname(__file__), 'targets.txt')
TESTDATA_LDA = os.path.join(os.path.dirname(__file__), 'lda_model.sav')
TESTDATA_PCA = os.path.join(os.path.dirname(__file__), 'pca_model.sav')

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

        results = spep.load('pudu_test_results.txt')

        def pf(X):
            X = X[0,:,:,0]
            return lda.predict_proba(pca.transform(X))[0]

        imp = pudu.pudu(x, y, pf)

        imp.importance(delta=1, window=100, calc='absolute')
        imp.speed(delta=1, window=100, calc='absolute')
        imp.synergy(delta=1, inspect=3, window=100, calc='absolute')
        
        self.assertEqual((imp.imp[0,0,:,0] == results[0]).all(), True)
        self.assertEqual((imp.imp_norm[0,0,:,0] == results[1]).all(), True)

        self.assertEqual((imp.spe[0,0,:,0] == results[2]).all(), True)
        self.assertEqual((imp.spe_norm[0,0,:,0] == results[3]).all(), True)

        self.assertEqual((imp.syn[0,0,:,0] == results[4]).all(), True)
        self.assertEqual((imp.syn_norm[0,0,:,0] == results[5]).all(), True)
