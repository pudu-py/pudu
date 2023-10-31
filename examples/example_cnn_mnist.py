import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt

from pudu import pudu, plots

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
model.summary()
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0], "| Test accuracy:", score[1])


### PUDU ###
# Input should be 4d: (batch * rows * columns * depth)
# This is a classification problem, thus the output should be the 
# probability by category: [prob_1, prob_2, ..., prob_c]
def cnn2d_prob(X):
    X = X[0,:,:,:]
    return model.predict(np.array([X, X]), verbose=0)[0] # verbose 0 is important!

# Dimention standarization for parameters
y = np.argmax(y_train[0])
x = np.expand_dims(x_train[0], 0)

# Build `pudu`, evaluate importance, and plot
imp = pudu.pudu(x, y, cnn2d_prob)

# in this case, 'window' is a tuple that indicates the width and height.
imp.importance(window=(3, 3), scope=None, padding='center')
plots.plot(imp.x, imp.imp_rel, axis=None, figsize=(7, 6), cmap='cool', title='Relative Importance',
            xlabel='', ylabel='')

# In this case, as there are many `0` values in the image, including some bias 
# can help us to visualize importance in those areas, since 0*delta = 0
imp.importance(window=(3, 3), scope=None, padding='center', bias=0.1)
plots.plot(imp.x, imp.imp_rel, axis=None, figsize=(7, 6), cmap='cool', title='Relative Importance - with bias',
            xlabel='', ylabel='')

### LIME ###
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb

# lime_image works with 3D RGB arrays. Our image is non-RGB. We transform it
# to RGB with gray2rgb which needs a 2D input.
image = x_train[0,:,:,0]
image = gray2rgb(image) 

# model.predict is not for RGB, so we have to wrap it so the input is RBG.
def cnn2d_prob(X):
    X = X[:, :, :, 0]
    X = X[:, :, :, np.newaxis]
    return model.predict(X, verbose=0)

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image, cnn2d_prob, batch_size=1, num_samples=1000)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)

plt.rc('font', size=15)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.figure(figsize=(6, 6))

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask), extent=[0, 28, 0, 28])

plt.title('LIME') 
plt.tight_layout()
plt.show()  

### GRAD CAM ###
import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


last_conv_layer_name = 'conv2d_5'
image = np.expand_dims(x_train[0], axis=0)

# Remove last layer's softmax
model.layers[-1].activation = None

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(image, model, last_conv_layer_name)



plt.rc('font', size=15)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.figure(figsize=(7, 6))

plt.imshow(image[0,:,:,:], cmap='binary', alpha=1, extent=[0, 28, 0, 28])
plt.imshow(heatmap, cmap='jet', aspect="auto", alpha=0.75, extent=[0, 28, 0, 28])

plt.title('GRADCAM') 
plt.tight_layout()
plt.colorbar(pad=0.05)
plt.show()
