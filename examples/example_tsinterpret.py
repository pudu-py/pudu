import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle
import numpy as np 
import torch 
from ClassificationModels.CNN_T import ResNetBaseline, UCRDataset
from tslearn.datasets import UCR_UEA_datasets
import sklearn
import spectrapepper as spep
from tensorflow import keras
from keras.models import load_model

import os
import pickle
import numpy as np 
import torch 
from ClassificationModels.CNN_T import ResNetBaseline, UCRDataset,fit
import pandas as pd
import os 
from tslearn.datasets import UCR_UEA_datasets
import sklearn


# Scale images to the [0, 1] range
x = spep.load('examples/data/for_1d_cnn_c3.txt')
x = np.expand_dims(x, 2)
y = [3, 3, 3] # these are all class 3

# convert class vectors to binary class matrices
y = keras.utils.to_categorical(y)

# Load the model and test it
model = load_model('examples/data/1d_cnn.h5')

# model.eval()
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
int_mod=TSR(model, x.shape[-1],x.shape[-2], method='GRAD', \
    mode='feat')

item = np.array([x[0,:,:]])
label = int(np.argmax(y[0]))

exp=int_mod.explain(item, labels=label, TSR=True)

int_mod.plot(np.array([x[0,:,:]]), exp, figsize=(15, 15))

int_mod.plot(np.array([x[0,:,:]]), exp, heatmap=True)