from localreg import RBFnet, plot_corr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import spectrapepper as spep
from pudu import pudu
import numpy as np

# Load the file that contains the calculated areas from the spectras
ra = spep.load('data/areas.txt', transpose=True)

# Shuffleing and spearating into area type and train/test
ra0, ra1, ra2, ra3, voc = spep.shuffle([ra[0], ra[1], ra[2], ra[3], ra[4]])
ra0_te, ra1_te, ra2_te, ra3_te, voc_te = ra0[1494:], ra1[1494:], ra2[1494:], ra3[1494:], voc[1494:]
ra0_tr, ra1_tr, ra2_tr, ra3_tr, voc_tr = ra0[:1494], ra1[:1494], ra2[:1494], ra3[:1494], voc[:1494]

# Preparation, training, and testing of the RBNF
input = np.array([ra0_tr, ra1_tr, ra2_tr, ra3_tr]).T
z = voc_tr
net = RBFnet()
net.train(input, z, num=200, verbose=False)
z_hat = net.predict(input)
z_hat_te = net.predict(np.array([ra0_te, ra1_te, ra2_te, ra3_te]).T)
print('Train R2: ', round(r2_score(z, z_hat), 2), ' | Test R2: ', round(r2_score(voc_te, z_hat_te), 2))

# Plot the regression
plt.scatter(z_hat, z, s=1, c='green')
plt.scatter(z_hat_te, voc_tr, s=1, c='green')
plt.ylabel('Value')
plt.xlabel('Prediction')
plt.show()

### PUDU ###
# Wrap the probability function
def rbf_pred(X):
    X = X[0,0,:,0]
    return net.predict([X, X])[0]

# Formatting as (batch, rows, columns, depth)
x = input[0]
x = x[np.newaxis, np.newaxis, :, np.newaxis]
y = z[0]

# Build pudu and evaluate importance
imp = pudu.pudu(x, y, rbf_pred)
imp.importance(delta=0.1, window=1, scope=(0, 4), calc='absolute')
imp.plot(imp.x, imp.imp, title="Importance", font_size=25, show_data=False)

# Plot the ouput
axis = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
ticks = ['', 'A1', '', 'A2', '', 'A3', '', 'A4', '']
imp.plot(imp.x, imp.imp, axis=axis, xticks=ticks, title="Importance", 
    font_size=25, show_data=False)
