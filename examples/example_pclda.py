"""
## EXPLANATION START ##

This function calculates the hypotenuse of a right triangle.

The calculation is based on the Pythagorean theorem:

.. math::

   c = \sqrt{a^2 + b^2}

where:
- :math:`c` is the length of the hypotenuse,
- :math:`a` and :math:`b` are the lengths of the other two sides.

## EXPLANATION END ##
"""

import matplotlib.pyplot as plt
from lime import lime_tabular
import spectrapepper as spep
import numpy as np
import pickle
import lime

# from pudu import pudu, plots
import pudu7 as pudu
import plots

# Load features (spectra) and targets (open circuit coltage, voc)
features = spep.load('data/features.txt')
targets = spep.load('data/targets.txt', transpose=True)[2]

# Load pre-trained LDA and PCA models
lda = pickle.load(open('data/lda_model.sav', 'rb'))
pca = pickle.load(open('data/pca_model.sav', 'rb'))

### LIME ###
# First we try LIME. then we try pudu and see the difference.
# We need to wrap the probability function
def pcalda_proba(X):
    X = pca.transform(X)
    return lda.predict_proba(X)

# Feature names and categorical features in the correct format
fn = [str(i) for i in range(len(features[100]))]
cf = [i for i in range(len(features[100]))]

# Make explainer and evaluate an instance
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(features), 
    mode='classification', feature_names=fn, categorical_features=cf, verbose=False)
exp = explainer.explain_instance(features[10], pcalda_proba, 
                                 num_features=len(fn), num_samples=1000)

# Reformat the output so it is in order to plot over the feature
e = exp.as_list()
lm = [0 for _ in range(1536)]
for i in e:
    lm[int(str.split(i[0], '=')[0])] = i[1]

# Plot the result with the evaluated feature
plt.rc('font', size=15)
plt.figure(figsize=(14, 4))
plt.imshow(np.expand_dims(lm, 0), cmap='Greens', aspect="auto", interpolation='nearest', 
    extent=[0, 1536, min(features[100]), max(features[100])], alpha=1)
plt.plot(features[100], 'k')
plt.colorbar()
plt.title('Lime') 
plt.xlabel('Feature')
plt.ylabel('Intensity')
plt.yticks([])
plt.show()


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

# Evaluate `importance` for all features
imp.importance(delta=0.1, window=1, mode='bidirectional')
plots.plot(imp.x, imp.imp_norm, title="Importance", yticks=[], font_size=15)

# Single pixels might be irrelevant for spectroscopy. We can group features
# to evaluate together.
imp.importance(delta=0.1, window=50, mode='bidirectional')
plots.plot(imp.x, imp.imp, title="Importance", yticks=[], font_size=15)

# We can see how fast would the classification change according to
# the change in the features.
imp.speed(delta=0.1, window=50, mode='bidirectional')
plots.plot(imp.x, imp.spe, title="Speed", yticks=[], font_size=15)

# Finally we evaluate how different changes complement each other.
# We want ot evaluate the main Raman peak (aprox. 3rd position) and see
# its synergy with the rest of the data
imp.synergy(delta=0.1, inspect=3, window=50, mode='bidirectional')
plots.plot(imp.x, imp.syn, title="Synergy", yticks=[], font_size=15)
