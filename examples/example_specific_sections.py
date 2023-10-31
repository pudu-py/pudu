import spectrapepper as spep
import numpy as np
import pickle

from pudu import pudu, plots


# Load features (spectra) and targets (open circuit coltage, voc)
features = spep.load('data/features.txt')
targets = spep.load('data/targets.txt', transpose=True)[2]

# Load pre-trained LDA and PCA models
lda = pickle.load(open('data/lda_model.sav', 'rb'))
pca = pickle.load(open('data/pca_model.sav', 'rb'))


### PUDU ###
# Select x (feature) and respective y (target)
x = features[100]
x_len = len(x)
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

# To evaluate more specific areas, we can do ir one-by-one and the
# put them all together in a vector for plotting and/or normalizing.
# This can be done for `importance` and `speed` only.

# For this, we first define the areas of interest
areas = [[170, 200], [225, 250], [250, 290], [300, 330]]

# In a loop we evaluate them individially. We make sure that `window`
# and `scope` are equal so all the area is evaluated. The results are
# saved in `custom`.
custom = np.zeros(x_len)
for i in areas:
    imp.importance(window=int(i[1]-i[0]), scope=(i[0], i[1]))
    custom[imp.imp[0, 0, :, 0] != 0] = imp.imp[0, 0, imp.imp[0, 0, :, 0] != 0, 0]

custom = custom[np.newaxis, np.newaxis, :, np.newaxis]
plots.plot(imp.x, custom, title="Importance", yticks=[], font_size=15, cmap='jet')

# Repeat the same for `speed`.
custom = np.zeros(x_len)
for i in areas:
    imp.speed(window=int(i[1]-i[0]), scope=(i[0], i[1]))
    custom[imp.spe[0, 0, :, 0] != 0] = imp.spe[0, 0, imp.spe[0, 0, :, 0] != 0, 0]

custom = custom[np.newaxis, np.newaxis, :, np.newaxis]
plots.plot(imp.x, custom, title="Speed", yticks=[], font_size=15, cmap='jet')
