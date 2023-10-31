import numpy as np
from tensorflow import keras
from keras.models import load_model

from pudu import pudu, plots
from pudu import perturbation as ptn


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load the model and test it
model = load_model('data/mnist_class.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0], "| Test accuracy:", score[1])
model.summary()

### PUDU ###
# Input should be 4d: (batch * rows * columns * depth)
# This is a classification problem, thus the output should be the 
# probability by category: [prob_1, prob_2, ..., prob_c]
def cnn2d_prob(X):
    X = X[0,:,:,:]
    return model.predict(np.array([X, X]), verbose=0)[0] # verbose 0 is important!

# Dimention standarization for parameters
sample = 10
y = np.argmax(y_train[sample])
x = np.expand_dims(x_train[sample], 0)

# Build `pudu`, evaluate importance, and plot
imp = pudu.pudu(x, y, cnn2d_prob, model)

# First we check importance
imp.importance(window=(3, 3), perturbation=ptn.Positive(delta=0.1), bias=0.1)
plots.plot(imp.x, imp.imp_rel, axis=None, figsize=(6, 6), cmap='cool')

# Now we explore the unit activations observed in layer 2 (last conv. layer)
# We only consider the top 0.5% of the values as activated with `p=0.005`.
# Negative values indicate that less units are being activated.
imp.reactivations(layer=2, slope=0, p=0.005, window=(5, 5), perturbation=ptn.Positive(delta=0.1))
plots.plot(imp.x, imp.lac, axis=None, figsize=(6, 6), cmap='cool', 
            title='Nº of unit activations in layer 0')

# There could be units that activate more frequently with changes in specific areas
# We can explore that too using the information obtained from 'activations'
# and generate a visual (and/or printed) report
plots.feature_report(imp.fac, plot=True, print_report=False, show_top=10)
plots.unit_report(imp.uac, plot=True, print_report=False, show_top=20, font_size=12)

# To better visualize the coordinates, we can better check with 'preview'
imp.preview(window=(3, 3), figsize=(6, 6), yticks=None)

# We can extract more infomration when using several images and `relatable`. 
# This will see, after applying `activations` to N images, what units activate
# with what features the most. 
N = 10 # The more the better, 10 to keep computational times "low"
y = [np.argmax(i) for i in y_train[:N]]
x = x_train[:N]
imp = pudu.pudu(x, y, cnn2d_prob, model) # we need to build another pudu
imp.relatable(layer=2, slope=0, p=0.005, window=(3, 3), perturbation=ptn.Positive(delta=0.1))
idx, cnt, cor = imp.icc # this outputs the unit index, activation counts, and position of the feature
plots.relate_report(idx, cnt, cor, plot=True, print_report=True, show_top=30, font_size=10, rot_ticks=90, sort=True)
