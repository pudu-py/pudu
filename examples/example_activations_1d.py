import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import spectrapepper as spep
# from pudu import pudu, plots
import pudu7 as pudu
import plots

# Scale images to the [0, 1] range
x = spep.load('for_1d_cnn_c3.txt')
x = np.expand_dims(x, 2)
y = [3, 3, 3] # these are all class 3

# convert class vectors to binary class matrices
y = keras.utils.to_categorical(y)

# Load the model and test it
model = load_model('0_526_model_(80;79).h5')
score = model.evaluate(x, y, verbose=0)
print("Test loss:", score[0], "| Test accuracy:", score[1])
model.summary()

### PUDU ###
# Input should be 4d: (batch * rows * columns * depth)
# This is a classification problem, thus the output should be the 
# probability by category: [prob_1, prob_2, ..., prob_c]
def cnn1d_prob(X):
    X = X[0,:,:,:]
    return model.predict(np.array(X), verbose=0)[0] # verbose 0 is important!

# Dimention standarization for parameters
y0 = np.argmax(y[0])
x0 = np.expand_dims(np.expand_dims(x[0], 0), 1)

# Build `pudu`, evaluate importance, and plot
imp = pudu.pudu(x0, y0, cnn1d_prob, model)

# First we check importance
imp.importance(window=150, mode='positive', delta=0.1)
plots.plot(imp.x, imp.imp, axis=None, figsize=(10, 5), cmap='cool')

# Now we explore the unit activations observed in layer 4 (last conv. layer)
# We only consider the top 0.5% of the values as activated with `p=0.005`.
# Negative values indicate that less units are being activated.
imp.activations(layer=4, slope=0, p=0.0025, window=150, mode='positive', delta=0.1)
plots.plot(imp.x, imp.lac, axis=None, figsize=(10, 4), cmap='cool', 
            title='Nº of unit activations in layer 0')

# There could be units that activate more frequently with changes in specific areas
# We can explore that too using the information obtained from 'activations'
# and generate a visual (and/or printed) report
plots.feature_report(imp.fac, plot=True, print_report=False, show_top=10)
plots.unit_report(imp.uac, plot=True, print_report=False, show_top=10)

# To better visualize the coordinates, we can better check with 'preview'
imp.preview(window=150, figsize=(10, 4), yticks=None, bold=True)

# We can extract more infomration when using several data points and `relatable`. 
# This will check, after applying `activations` to N images, what units activate
# with what feature changes the most. Then, for N=3
# Change the shape of the array
x = x.reshape(3, 1, 14730, 1)
y = [np.argmax(i) for i in y]

imp = pudu.pudu(x, y, cnn1d_prob, model) # we need to build another pudu
imp.relatable(layer=2, slope=0, p=0.0025, window=150, mode='positive', delta=0.1)
idx, cnt, cor = imp.icc # this outputs the unit index, activation counts, and position of the feature
plots.relate_report(idx, cnt, cor, plot=True, show_top=20, font_size=10, rot_ticks=45, sort=True)