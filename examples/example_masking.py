import matplotlib.pyplot as plt
from lime import lime_tabular
import spectrapepper as spep
import numpy as np
import pickle
import lime

from pudu import pudu, plots
from pudu import masks as msk

# Masking allows you to cover your featuresof analysis. It serves the
# same purpose of `scope` but it works ni a structured and patterned way.
# This example shows the use of `mask` options available.

# Load features (spectra) and targets (open circuit coltage, voc)
features = spep.load('data/features.txt')
targets = spep.load('data/targets.txt', transpose=True)[2]

# Load pre-trained LDA and PCA models
lda = pickle.load(open('data/lda_model.sav', 'rb'))
pca = pickle.load(open('data/pca_model.sav', 'rb'))

### PUDU ###
# Select x (feature) and respective y (target)
x = features[100]
x = x[np.newaxis, np.newaxis, :, np.newaxis]
y = targets[100]

# Input should be 4d: (batch * rows * columns * depth)
# This is a classification problem, thus the output should be the 
# probability by category: [prob_1, prob_2, ..., prob_c]
def pf(X):
    X = X[0,:,:,0]
    return lda.predict_proba(pca.transform(X))[0]

# Build pudu
imp = pudu.pudu(x, y, pf)

# Evaluate `importance` with no mask first, then 'everyother' and 'random'
masks = [msk.All(), msk.EveryOther(), msk.RandomMask()]
m_names = ['All', 'EveryOther', 'RandomMask']
for i,j in zip(masks, m_names):
    imp.importance(window=100, absolute=False, mask=i)
    plots.plot(imp.x, imp.imp, title="Importance - "+j, yticks=[], font_size=15, cmap='jet')
