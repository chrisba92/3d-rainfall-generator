import pandas as pd
import numpy as np
import copy
from scipy import special
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

# =============================================================================
# Functions in script
# =============================================================================
def mult_exp(x, a, l):
    r"""Mulit-exponential function
    
    Parameters
    ----------
    x : float
        Uniform random number
    a : float
        Ratio of rain events in the class and total number of rain events
    l : float
        Inverse mean of rainfall depth in the class
    """
    return a * (-(1/l) * np.log(1- x))

def is_pos_def(mat):
    r"""Test if the input matrix is positive definite.
    
    Parameters
    ----------
    mat : array_like
        Correlation matrix to be tested
    
    Returns
    ----------
    test_result : bool
        Returns True of correlation matrix is positive definite
        and False otherwise
    
        
    """
    return np.all(np.linalg.eigvals(mat) > 0)

def normal2uniform(rnd):
    r"""Convert standard normal distributed numbers (N(1, 0)) to uniform distributed numbers (U[0, 1]).
    
    Parameters
    ----------
    rnd : numpy ndarray
        ndarray of standard nomral distributed numbers to be converted.
        
    Returns
    ----------
    z : numpy ndarray
        Transformed random numbers
    """
    z = np.zeros(rnd.shape)
    for i, r in enumerate(rnd):
        z[i,:] = 0.5 * special.erfc(-r / np.sqrt(2))
        
    return z

def diagonalize(mat):
    r"""Diagonalize non-positive indefinte matrix following
        the procedure from ....
    
    Parameters
    ----------
    mat : numpy ndarray
        Non-definite matrix that needs to be diagonlized into a
        positive definite one
        
    Returns
    ----------
    cr_ new : new positive definite correlation matrix
    """
    # Calculate eigenvalues and vectors
    d, m = np.linalg.eig(mat)
    d = np.diag(d)
    
    # Replace negative eigenvalues
    d[d<0] = 1e-7
    
    # Perform diagonalzation
    cr = m @ d @ m.T
    
    # Normalize matrix
    diag_cr = np.diag(cr)
    diag_cr = np.reshape(diag_cr, (diag_cr.size, 1))
    cr_new = cr / np.sqrt(diag_cr @ diag_cr.T)
    
    return cr_new

def correlated_rnd(corr_mat, rnd):
    r"""Creation of correlated random numbers using cholesky factorization
    
    Parameters
    ----------
    corr_mat : numpy ndarray
        Correlation matrix used to create the correlated random numbers.
    rnd : numpy ndarray
        ndarray of random numbers
        
    Returns
    ----------
    z : ndarray of the correlated random numbers
    """
    # Compute the (lower) Cholesky decomposition matrix
    chol = cholesky(corr_mat, lower=True)
    
    # Generate 3 series of normally distributed (Gaussian) numbers
    ans = chol @ rnd

    z = normal2uniform(ans)
    return z

def execute_markov(m, n, rnd, markov_models, names):
    r"""Feed correlated random numbers through a markov chain
    
    Parameters
    ----------
    m : int
        Number of measuring rainfall stations.
    n : int
        Number of sequences to simulate.
    rnd : numpy ndarray
        Matrix containg correlated uniform random numbers.
    markov_models : dcit
        Dictonary containg the markov chains for each station.
    names : array_like
        list or np array containg the given name to the rainfall stations.

    Returns
    -------
    seq : numpy ndarray
        Matrix with the state at each timestep for each station.

    """
    # Feed the random numbers through a markov process
    seq = np.zeros((m, n))
    seq[-1,0] = 1
    
    for i in range(1, n):
        pre_seq = seq[:,i-1]
        probs_full = rnd[:,i]
        
        for j in range(m):
            trans = markov_models[names[j]]['trans']
            if pre_seq[j] == 0:
                pc = trans[0,0]
            else:
                pc = trans[1,0]
            
            if probs_full[j] <= pc:
                seq[j,i] = 0
            else:
                seq[j,i] = 1
    return seq

def calc_occindex(occurence, m, occ_corr_org):
    r"""Calculate occurence for each station at each day with rain
    
    Parameters
    ----------
    occurence : numpy ndarray
        Occurence array with info about if a station have measured rainfall or not.
    m : int
        DESCRIPTION.
    occ_corr_org : TYPE
        DESCRIPTION.

    Returns
    -------
    occ_index : TYPE
        DESCRIPTION.

    """
    # Initialize the occurence index array
    occ_index = np.zeros(occurence.shape)
    
    for i in range(m):
        # Find all days with rain for the current station
        ids = np.where(occurence[:,i]>0)[0]
        
        # Look up the the current stations correlation with the other stations
        c = copy.copy(occ_corr_org[i,:])
        c = np.delete(c, i)
        
        # Unit vector - needed for the calucations
        u = np.ones(c.shape)
        
        # Go through each rainy day and calculate occurence index
        for id_ in ids:
            o = copy.copy(occurence[id_,:])
            o = np.delete(o, i)
            
            km = np.dot(o,c) / np.dot(u, c)
            
            occ_index[id_, i] = km
    
    return occ_index
# =============================================================================
# Load and process the rainfall data
# =============================================================================
print('Loading the daily rainfall data...', end='')

# Set the rainfall file to be loaded - NB! Should be a file of daily rainfall amounts
filename = 'daily_rainfall.csv'

# Load the rainfall file into a pandas dataframe
df = pd.read_csv(filename, 
                 index_col=0, 
                 parse_dates=['Dates'])

# Extract the values to a np array
daily = df.values

# Store column names for later use
names = df.columns.values

# Filter out low values Anything lower than the resolution
#daily[daily<0.3] = 0

# Get correlation matrix of the rainfall amounts
rainfall_corr_obs = df.corr()

# Transform into occurence array - 1 means wet day, 0 means dry day
occurence = copy.copy(daily)
occurence[occurence>0] = 1

# Determine the correlation matrix of the occurences
occ_df = pd.DataFrame(data=occurence, columns=names)
occ_corr_org = occ_df.corr().values

# Print out status of the data processing
print('Done!')

# =============================================================================
# Fit markov chain to each of the gauges
# =============================================================================
print('Fitting a markov chain to each of the rain gauges...', end='')
markov_models = {}
for j in range(daily.shape[1]):
    name = names[j]
    
    # Setup markov model
    markov_models[name] = {}
    markov_models[name][0] = [] # Dry state
    markov_models[name][1] = [] # Wet state
    
    # Get sequences for the current rain gauge
    seq = occurence[:,j]
    
    # Extract dependt occurences
    for i in range(seq.size-1):
        markov_models[name][seq[i]].append(seq[i+1])
        
    # Create transistion matrix
    trans = np.zeros((2,2))
    trans[0,0] = np.sum(np.array(markov_models[name][0])==0) / len(markov_models[name][0])
    trans[0,1] = np.sum(np.array(markov_models[name][0])==1) / len(markov_models[name][0])
    
    trans[1,0] = np.sum(np.array(markov_models[name][1])==0) / len(markov_models[name][1])
    trans[1,1] = np.sum(np.array(markov_models[name][1])==1) / len(markov_models[name][1])
    
    # Save the model
    markov_models[name]['trans'] = trans

print('Done!')

# =============================================================================
# Autodetermination of new correlation matrix
# =============================================================================
print('Determining new correlation matrix...')
# Set random seed
np.random.seed(1234)
    
# Total number of sequences to model
n = occurence.shape[0]

# Set number of stations
m = occurence.shape[1]

# Copy original occurence array - Using copy to avoid pointer issues
occ_corr = copy.copy(occ_corr_org)

fitness = []
n_sim = 1000
rnd_ = np.random.normal(0.0, 1.0, size=(m, n))
last_avg = None
for p in range(n_sim):
    # Test if the current correlation matrix is positive definite
    if is_pos_def(occ_corr) == False:
        # Diagonlize the matrix if it is not definite
        occ_corr = diagonalize(occ_corr)
    
    # Create correlated random, uniform, numbers
    rnd = correlated_rnd(occ_corr, rnd_)
    
    # Feed the random numbers through a markov process
    seq = execute_markov(m, n, rnd, markov_models, names)
                
    # Get correlation
    occ_corr_temp = pd.DataFrame(data=seq.T, columns=names).corr()
    
    # Get the difference between original correlation matrix and new one
    dif = occ_corr_org - occ_corr_temp.values
    
    # Add the difference to the correlation matrix
    occ_corr = occ_corr + 0.1*dif
    
    # Log the score of the solution scheme
    fitness.append(np.sum(np.abs(dif)))
    
    # After 10 runs, test if the scheme converged
    if p>=10:
        # Get the last 10 fitness scores
        temp = np.array(fitness[-10:])
        
        # Calculate the change between each iteration
        change = np.abs(temp[1:] - temp[:-1])
        
        # Get the average change
        avg = np.mean(change)
        
        # Test if scheme have converged
        if last_avg is not None and np.isclose(avg, last_avg, atol=1e-4) and fitness[-1]<1:
            print('\tsolution scheme converged!')
            print(f'\tthe score ended up at {fitness[-1]:.3f}')
            break
        
        last_avg = avg

if p==n_sim-1:
    print('\tmaximum number of iteration hit...')
    print(f'\tcurrent score is {fitness[-1]}')
    
# Diagonlize the final matrix, if it is not positive definite
if is_pos_def(occ_corr) == False:
    # Diagonlize the matrix if it is not definite
    occ_corr = diagonalize(occ_corr)
    
# =============================================================================
#  Establish link between occurence index and precip. amount
# =============================================================================
print('Creating model for daily rainfall amounts...')

# Calculate occurence indexs for each station
print('\tcalculating occurence index..')
occ_index = calc_occindex(occurence, m, occ_corr_org)

# Create season array
seasons = [[12, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
          [9, 10, 11]]

season_name = ['DJF', 'MAM', 'JJA', 'SON']

# Initialze multip exponential model
print('\tcreating multi-exponential model...')
alpha_list = []
lambda_list = []
retbin_list = []
for i in range(occ_index.shape[1]):
    n_class = 6
    
    # Categorize the data into 6(4) different classes - FIGURE OUT A WAY TO AUTO DETERMINE THE NCLASS
    df_temp = pd.DataFrame(data=occ_index[:, i], columns=['occ_index'])
    df_temp = df_temp.drop(df_temp[occurence[:,i]==0].index)
    _, bins = pd.qcut(df_temp['occ_index'], n_class, 
                                     duplicates='drop', retbins=True)
    n_class = bins.size-1
    df_temp['class'] = pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(n_class), include_lowest=True)
     
    # Save the class definition for the model
    retbin_list.append(bins)
    
    # Pandas messes me up a bit, so have to manually reduce the number of classes
    #n_class -= 2
    
    # Locate rainfall amount for each class and month
    month_class_mean = {}
    for k, (val, class_) in enumerate(zip(df_temp['occ_index'], df_temp['class'])):
        if val>0: # Skip if no rain have occured
            if class_ not in month_class_mean: # If class haven't been added to the dict, initialized it
                month_class_mean[class_] = {}
            
            # Extract the month
            month = df.index[k].month 
            if month not in month_class_mean[class_]: # Add the month to the dict, if it does not exist
                month_class_mean[class_][month] = []
            
            month_class_mean[class_][month].append(df.iloc[df_temp.index[k], i])
   
    # Go through each class, and calculate the mean and save it for each season
    result = {}
    for key in range(n_class):
        class_values = month_class_mean[key]
        result[key] = {}
        for m, season in enumerate(seasons):
            result[key][season_name[m]] = []
            temp = [] 
            for s in season:
                if s in class_values:
                    temp.extend(class_values[s])
            season_average = np.mean(temp)
            result[key][season_name[m]].append(season_average)
    lambda_list.append(result)
            
    # Build multi-exponential distribution
    alpha = []
    for class_ in range(n_class):
        temp = []
        for month in month_class_mean[class_]:
            temp.append(len(month_class_mean[class_][month]))
        alpha.append(np.sum(temp))
        
    alpha = np.array(alpha) / np.sum(alpha)
    alpha_list.append(alpha)
    
#%%
# =============================================================================
# Let's create some rainfall!!
# =============================================================================
# Initialze some stuff!
# Set random seed
np.random.seed(1234)
    
# Total number of sequences to model
n = occurence.shape[0]

# Set number of stations
m = occurence.shape[1]

# Create time array
time_array = pd.date_range(start=df.index[0], periods=n)

season_dict = {12 : 'DJF', 1 : 'DJF', 2 : 'DJF',
               3 : 'MAM', 4 : 'MAM', 5 : 'MAM',
               6: 'JJA', 7: 'JJA', 8: 'JJA',
               9: 'SON', 10: 'SON',  11: 'SON'}
   
## Step 1 - Create, correlated, occurence array ##    
# Generate standard normal random numbers
rnd_ = np.random.normal(0.0, 1.0, size=(m, n))

# Create correlated random, uniform, numbers
rnd = correlated_rnd(occ_corr, rnd_)

# Simulate occurences
seq = execute_markov(m, n, rnd, markov_models, names)
#seq = copy.copy(occurence.T)

## Step 2 - Calculate occurence index for each station ##
occ_index_new = calc_occindex(seq.T, m, occ_corr_org)

## Step 3 - Determine rainfall amounts! ##
rainfall = np.zeros((n,m))
x_array = np.random.rand(n)
for i in range(occ_index_new.shape[1]):
    # Unpack model parameters
    bins   = retbin_list[i]   # Class definiton
    result = lambda_list[i]   # Lambda values for the mutl exp function
    alpha  =  alpha_list[i]   # Alpha values for the mult exp function
    
    # Categorize the data into the 4 different class'
    df_temp = pd.DataFrame(data=occ_index_new[:, i], columns=['occ_index'])
    df_temp = df_temp.drop(df_temp[seq.T[:,i]==0].index)
    df_temp['class']= pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(bins.size-1), include_lowest=True)
    
    for k, class_ in enumerate(df_temp['class']):
        # Determine season for rainfall occurence
        season = season_dict[time_array[k].month]
        
        # Determine rainfall amount
        x = x_array[df_temp.index[k]]
        temp_rain = []
        for j in range(bins.size-1):
            a = alpha[j-1]
            l = 1 / result[j][season][0]
            temp_rain.append(mult_exp(x, a, l))
        
        rainfall[df_temp.index[k], i] = np.sum(temp_rain)
            
#%%        
# =============================================================================
# Perform diagnostics of the rainfall simulator!
# =============================================================================
## Process the simulated rainfall ##
# Add the simulated rainfall to a dataframe
df_rainfall = pd.DataFrame(data=rainfall, index=time_array, columns=names)

# Calculate the correlation of the rainfall amounts
rainfall_corr_sim = df_rainfall.corr().values

# Calculate the simulated occurence correlation
df_occ_sim = pd.DataFrame(data=seq.T, columns=names)  
occ_sim_corr = df_occ_sim.corr().values
# ## Compare occurence correlation ##
# org = np.reshape(occ_corr_org, (occ_corr_org.size,))
# org[org==1] = np.nan

# sim = np.reshape(occ_corr, (occ_corr.size,))
# sim[sim==1] = np.nan

# sim2 = np.reshape(occ_corr_temp.values, (occ_corr_temp.values.size,))
# sim2[sim2==1] = np.nan

# fig, ax = plt.subplots()
# ax.scatter(org,sim2)
# plt.title('Generated')
# ax.set(xlim=(0.4, 1), ylim=(0.4, 1))
# diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
# ax.set(xlim=(0.4, 1), ylim=(0.4, 1))

# ax.set_xlabel('Observed correlation [-]')
# ax.set_ylabel('Simulated correlation [-]')
# ax.set_title('Occurence correlation')
# ax.grid('Major')


































