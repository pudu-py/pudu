import matplotlib.pyplot as plt
from lime import lime_tabular
import spectrapepper as spep
import numpy as np
import pickle
import lime

# from pudu import pudu, plots
import pudu7 as pudu
import plots

# Other examples are with `bidirectional` or `positive` mode, that
# is an average of changing positevly and neatively the feature by 
# `delta` times. Depending on the nature of your features, you can
# diferent modes that will yield different results

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

# Evaluate `importance` with 'bidirectional', 'positive', 'negative', and 'relu'
# Note that some of these work only with `w=1`, like `relu` and `leakyrelu`
pert = ['bidirectional', 'positive', 'negative', 'relu']
for i in pert:
    imp.importance(delta=0.1, window=1, mode=i, absolute=True)
    plots.plot(imp.x, imp.imp, title="Importance - "+i, yticks=[], font_size=15)