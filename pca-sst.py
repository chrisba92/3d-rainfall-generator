from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

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

def mse(y, yh):
     return np.square(y - yh).mean()
 
def pca_autoencode(pca, data):
    # Test the error for the test data
    reduced_data = pca.transform(data)
    approx = pca.inverse_transform(reduced_data)
    
    error = []
    for i in range(data.shape[0]):
        error.append(mse(data[i, :], approx[i,:]))
        
    return error
    

# =============================================================================
# Load the data
# =============================================================================
data = np.loadtxt('rainfall_generator.txt')
scaler = MinMaxScaler()
data_standard = scaler.fit_transform(data)

# Spilt the data set
train_data, test_data = random_sample(data_standard, 0.9)

# =============================================================================
# Train the PCA on the training data
# =============================================================================
pca = PCA(.95)

# Perform dimensionality reduction
reduced_data = pca.fit(train_data)

error = pca_autoencode(pca, train_data)
np.savetxt('error.dat', error)

hist, bins = np.histogram(error, bins='fd')
test = np.cumsum(hist)
test = test / len(error)
thresh = bins[np.where(test>0.95)[0][0]]
# print(np.histogram(error, bins='fd'))
# print(np.max(error))
    
# =============================================================================
# Load simulated rainfall data
# =============================================================================
scaler = MinMaxScaler()
rainfall = np.loadtxt('mult_gets_rainfall.dat')
test = np.where(np.any(rainfall, axis=1)==False)[0]
rainfall = np.delete(rainfall, test, axis=0)
rainfall[rainfall==0] = -999
rainfall_reduced = scaler.fit_transform(rainfall)

# Perform PCA deconstruction and reconstruction
error_sim = pca_autoencode(pca, rainfall_reduced)
test2 = error_sim<=thresh
print(np.sum(error_sim<=thresh))

# print(np.histogram(error_sim, bins=bins))
# print(np.max(error_sim))




















    
