from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import log_loss

import numpy as np

import matplotlib.pyplot as plt

from copy import copy

import sys

# =============================================================================
# Functions
# =============================================================================
def random_sample(arr, frac):
    r"""
    ""

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    frac : TYPE
        DESCRIPTION.

    Returns
    -------
    train_data : TYPE
        DESCRIPTION.
    test_data : TYPE
        DESCRIPTION.

    """
    n = arr.shape[0]
    
    train_n = int(n * frac)
    
    x = np.arange(n)
    np.random.shuffle(x)
    
    train_data = arr[x[:train_n], :]
    test_data = arr[x[train_n:], :]
    
    return (train_data, test_data)


# =============================================================================
# User inputs
# =============================================================================
mulGETS = 'mult_gets_rainfall.dat'
sst = 'rainfall_generator.txt'

# =============================================================================
# Load training data
# =============================================================================
data = np.loadtxt(sst)
scaler = StandardScaler()
data_standard = scaler.fit_transform(data)

train_data = data_standard

# =============================================================================
# Common input
# =============================================================================
train_model = False
input_dim = train_data.shape[1]

epochs = 30
batch_size = 100

# =============================================================================
# Build the normal autoencoder
# =============================================================================
encoding_dim = 2
if train_model:
    # Set input layer
    ilayer = Input(shape=(input_dim,))
    
    # Set encoder layer
    encoder_simple_layer = Dense(encoding_dim, activation='relu')(ilayer)
    
    # Set the output layer - alsoe the decoder layer
    decoder_simple_layer = Dense(input_dim, activation='tanh')(encoder_simple_layer)
    
    # Create the model
    simple_autoencoder = Model(ilayer, decoder_simple_layer)
    
    # Compile the model
    simple_autoencoder.compile(loss='mse', optimizer='adam')
    
    # Train the model
    simple_autoencoder.fit(train_data, train_data,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)
    
    # Save the model
    simple_autoencoder.save('models/simple_autoencoder.h5')
else:
    simple_autoencoder = load_model('models/simple_autoencoder.h5')



# =============================================================================
# Build the deeply connected autoencoder
# =============================================================================
encoding_dim = 2
if train_model:
    # Set input layer
    ilayer = Input(shape=(input_dim,))
    
    # Set the first couple of hidden layers
    layer1 = Dense(7, activation='relu')(ilayer)
    layer2 = Dense(3, activation='relu')(layer1)
    
    # Set the encoding layer
    encoder_deeply_layer = Dense(encoding_dim, activation='relu')(layer2)
    
    # Set the last part of the decoding layers
    layer3 = Dense(3, activation='relu')(encoder_deeply_layer)
    layer4 = Dense(7, activation='relu')(layer3)
    decoder_deeply_layer = Dense(input_dim, activation='sigmoid')(layer4)
    
    # Create the model
    deep_autoencoder = Model(ilayer, decoder_deeply_layer)
    deep_autoencoder.compile(loss='mse', optimizer='adam')
    
    # Train the model
    deep_autoencoder.fit(train_data, train_data,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)
    
    # Save the model
    deep_autoencoder.save('models/deep_autoencoder.h5')
else:
    deep_autoencoder = load_model('models/deep_autoencoder.h5')

# =============================================================================
# Determine threshold value for both encoders
# =============================================================================
# Simple encoder
simple_data = copy(train_data)
simple_decoded = simple_autoencoder.predict(simple_data)
simple_error   = (np.square(simple_data - simple_decoded)).mean(axis=1)

error_sorted = np.sort(simple_error)
thresh_test = error_sorted[int((1-0.05/2)*error_sorted.size)]

# Seperate the errors in to bin
hist, bins = np.histogram(simple_error, bins='fd')

# Select threshold based on the 95-percentile
test = np.cumsum(hist)
test = test / len(simple_error)
thresh_simple = bins[np.where(test>0.95)[0][0]]
print(f'Threshold for the simple autoencoder, by binning, is chosen to be: {thresh_simple}')
print(f'Threshold for the simple autoencoder, by cdf, is chosen to be: {thresh_test}')

# Deep encoder
deep_data = copy(train_data)
deep_decoded = deep_autoencoder.predict(deep_data)
deep_error   = (np.square(deep_data - deep_decoded)).mean(axis=1)

# Seperate the errors in to bin
hist, bins = np.histogram(deep_error, bins='fd')

# Select threshold based on the 95-percentile
test = np.cumsum(hist)
test = test / len(deep_error)
thresh_deep = bins[np.where(test>0.9)[0][0]]
print(f'Threshold for the deep autoencoder is chosen to be: {thresh_deep}')

# =============================================================================
# Load the simulated rainfal data
# =============================================================================
#scaler = MinMaxScaler()
rainfall = np.loadtxt(mulGETS)
#rainfall = rainfall[~np.all(rainfall == 0, axis=1)]

rainfall_scaled = scaler.transform(rainfall)
np.savetxt('test_scaled.dat', rainfall_scaled, fmt='%.3f')

rainfall_simple = copy(rainfall_scaled) 
rainfall_simple = simple_autoencoder.predict(rainfall_scaled)

sim_error_simple = (np.square(rainfall_scaled - rainfall_simple)).mean(axis=1)
print(np.sum(sim_error_simple<=thresh_simple))
test = rainfall[sim_error_simple<=thresh_simple, :]
print(test)


rainfall_deep = copy(rainfall_scaled) 
rainfall_deep = deep_autoencoder.predict(rainfall_scaled)

sim_error_deep = (np.square(rainfall_scaled - rainfall_deep)).mean(axis=1)
print(np.sum(sim_error_deep<=thresh_deep))
test = rainfall[sim_error_deep<=thresh_deep, :]
print(test)
np.savetxt('new_rainfall.dat', test, fmt='%.3f')
# error_sim = []
# for i in range(rainfall_reduced.shape[0]):
#     error_sim.append(mse(rainfall_scaled[i, :], rainfall_reduced[i,:]))
    
# hist, _ = np.histogram(error_sim, bins=bins)
# # print(hist)
# print(np.sum(error_sim<=thresh))
# np.savetxt('comparabledays.dat', error_sim<=thresh)
        






























