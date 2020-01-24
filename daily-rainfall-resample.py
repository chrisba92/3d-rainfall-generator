import pandas as pd
import numpy as np
import copy
from scipy import special
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import sys
import os

# =============================================================================
# Functions in script
# =============================================================================
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def rainfall_stat(data, timeres):
    '''
        Function to calculate selected statistics of the rainfall timeseries
        
        =====
        Input
        =====
        df       : Pandas Dataframe, a data frame containing the rainfall intesities for each timestep - The index column must be a datetime format of sorts
        timesres : int or float, temporal resolution of the timeseries, in minutes
        
    '''
    
    # Calculate statistics
    rainstats = {}
    
    # Calculate rain amount in every time step
    IntSum = data.Int / 60 * timeres
    data['IntSum'] = IntSum
    
    # Annual precipitation
    ap = data.IntSum.groupby(lambda x: x.year).sum()
    
    # Save stats
    rainstats['ap'] = [ap.mean(), ap.std()]
    
    # Seasonal precipitation
    test = data.IntSum.resample('M').sum().to_frame()
    
    month = data.IntSum.resample('M').sum()
    month_mean = month.groupby(lambda x: x.month).mean()
    month_var = month.groupby(lambda x: x.month).var()
    rainstats['month'] = [month_mean.values, month_var.values]
    
    # Create a Seasonal Dictionary that will map months to seasons
    SeasonDict = {11: 'Autumn', 12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', \
    8: 'Summer', 9: 'Autumn', 10: 'Autumn'}
    
    # Call the function with the groupby operation. 
    sp = test.IntSum.groupby([lambda x: x.year, lambda x: SeasonDict[x.month]]).sum()
    sp = sp.to_frame()
        
    labels = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    sp_var = []
    for label in labels:
        sp_var.append(sp.iloc[sp.index.get_level_values(1).str.contains(label)].var().values)
        
    rainstats['sp_var'] = [sp_var]
    
    
    # Winter 
    rainstats['spwi'] = [sp.iloc[sp.index.get_level_values(1).str.contains('Winter')].mean().values[0], sp.iloc[sp.index.get_level_values(1).str.contains('Winter')].std().values[0]]
    
    # Autumn
    rainstats['spau'] = [sp.iloc[sp.index.get_level_values(1).str.contains('Autumn')].mean().values[0], sp.iloc[sp.index.get_level_values(1).str.contains('Autumn')].std().values[0]]
    
    # Summer
    rainstats['spsu'] = [sp.iloc[sp.index.get_level_values(1).str.contains('Summer')].mean().values[0], sp.iloc[sp.index.get_level_values(1).str.contains('Summer')].std().values[0]]
    
    # Spring
    rainstats['spsp'] = [sp.iloc[sp.index.get_level_values(1).str.contains('Spring')].mean().values[0], sp.iloc[sp.index.get_level_values(1).str.contains('Spring')].std().values[0]]
            
    return rainstats


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

def execute_markov(m, n, rnd, markov_models, names, seasonal=False, time_array=None):
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
            if seasonal==False:
                if pre_seq[j] == 0:
                    pc = trans[0,0]
                else:
                    pc = trans[1,0]
                
                if probs_full[j] <= pc:
                    seq[j,i] = 0
                else:
                    seq[j,i] = 1
            else:
                season = season_dict[time_array[i].month]
                season_id = np.where(season_name == season)[0][0]
                if pre_seq[j] == 0:
                    pc = trans[season_id,0,0]
                else:
                    pc = trans[season_id,1,0]
                
                if probs_full[j] <= pc:
                    seq[j,i] = 0
                else:
                    seq[j,i] = 1
                
                
    return seq

def calc_occindex(occurence, m, occ_corr_org):
    r"""

    Parameters
    ----------
    occurence : TYPE
        DESCRIPTION.
    m : TYPE
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

def determine_corrmat(corr_mat, org_corr_mat, markov_models=None, time_array=None, corr_type='occurence', lr=0.1, n_sim=1000, crit=1, seasonal=False, plot_error=False, plot_dir=None):
    r"""Automatic determination of new correlation matrix

    Parameters
    ----------
    corr_mat : numpy ndarray
        Initial guess of correlation matrix.
    org_corr_mat : numpy ndarray
        Target correlation matrix - the output using corr_mat should
        be comparable with this.
    lr : float, optional
        Convergence critieon, low value will result in higher accuracy but
        lower convergence speed. The default is 0.1.
    n_sim : int, optional
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    corr_mat : numpy ndarray
        Converged, positive definite version of corr_mat

    """
    fitness = []
    rnd_ = np.random.normal(0.0, 1.0, size=(m, n))
    last_avg = None
    for p in range(n_sim):
        # Test if the current correlation matrix is positive definite
        if is_pos_def(corr_mat) == False:
            # Diagonlize the matrix if it is not definite
            corr_mat = diagonalize(corr_mat)
        
        # Create correlated random, uniform, numbers
        rnd = correlated_rnd(corr_mat, rnd_)
        
        if corr_type=='occurence':
            # Feed the random numbers through a markov process
            if seasonal==False:
                rnd = execute_markov(m, n, rnd, markov_models, names)
            else:
                rnd = execute_markov(m, n, rnd, markov_models, names, seasonal=True, time_array=time_array)
                    
        # Get correlation
        corr_temp = pd.DataFrame(data=rnd.T, columns=names).corr()
        
        # Get the difference between original correlation matrix and new one
        dif = org_corr_mat - corr_temp.values
        
        # Add the difference to the correlation matrix
        corr_mat = corr_mat + lr*dif
        
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
            if last_avg is not None and np.isclose(avg, last_avg, atol=1e-4) and fitness[-1]<crit:
                print('\tsolution scheme converged!')
                print(f'\tthe score ended up at {fitness[-1]:.3f}')
                break
            
            last_avg = avg
    
    if p==n_sim-1:
        print('\tmaximum number of iteration hit...')
        print(f'\tcurrent score is {fitness[-1]}')
    
    # Diagonlize the final matrix, if it is not positive definite
    if is_pos_def(corr_mat) == False:
        # Diagonlize the matrix if it is not definite
        corr_mat = diagonalize(corr_mat)
    
    if plot_error:
        fig, ax = plt.subplots()
        ax.plot(fitness)
        fig.savefig(plot_dir+'/corr_determine.png', dpi=300)
    return corr_mat

def fit_markov(names, occurence, seasonal=False):
    if seasonal==False:
        markov_models = {}
        for j in range(occurence.shape[1]):
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
    else:
        # Seasonal markov chains
        markov_models = {}
        for j in range(daily.shape[1]):
            name = names[j]
            
            # Initialize markov chain
            markov_models[name] = {}
            for season in season_name:
                markov_models[name][season] = {}
                markov_models[name][season][0] = []
                markov_models[name][season][1] = []
                
            # Get sequences for the current rain gauge
            seq = occurence[:,j]
            
            # Extract dependt occurences
            for i in range(seq.size-1):
                season = season_dict[df.index[i].month]
                markov_models[name][season][seq[i]].append(seq[i+1])
                
            # Create transistion matrix
            trans = np.zeros((4, 2,2))
            for i, season in enumerate(season_name):
                trans[i,0,0] = np.sum(np.array(markov_models[name][season][0])==0) / len(markov_models[name][season][0])
                trans[i,0,1] = np.sum(np.array(markov_models[name][season][0])==1) / len(markov_models[name][season][0])
                
                trans[i,1,0] = np.sum(np.array(markov_models[name][season][1])==0) / len(markov_models[name][season][1])
                trans[i,1,1] = np.sum(np.array(markov_models[name][season][1])==1) / len(markov_models[name][season][1])
                
            # Save the model
            markov_models[name]['trans'] = trans
    
    return markov_models

# =============================================================================
# User input values
# =============================================================================
# Set the rainfall file to be loaded - NB! Should be a file of daily rainfall amounts
filename = 'daily_rainfall.csv'

# Should diagnostic_plots be made?
diagnostic_plots = False

# Set current working directiory
cwd = os.getcwd()

# Set base path for diagnostic plots
plot_dir = 'diagnostic_plots/'
# Disable plotting
plt.ioff()


season_dict = {12 : 'DJF', 1 : 'DJF', 2 : 'DJF',
                3 : 'MAM', 4 : 'MAM', 5 : 'MAM',
                6: 'JJA', 7: 'JJA', 8: 'JJA',
                9: 'SON', 10: 'SON',  11: 'SON'}

season_name = np.array(sorted(['DJF', 'MAM', 'JJA', 'SON']))

# =============================================================================
# Load and process the rainfall data
# =============================================================================
print('Loading the daily rainfall data...', end='')
# Load the rainfall file into a pandas dataframe
df = pd.read_csv(filename, 
                 index_col=0, 
                 parse_dates=['Dates'])

# Total number of sequences to model
n = df.shape[0]

# Set number of stations
m = df.shape[1]

# Create time array
time_array = pd.date_range(start=df.index[0], periods=n)

# Extract the values to a np array
daily = df.values

# Store column names for later use
names = df.columns.values

# Create folders for each station - used for diagnostic plots
for name in names:
    # Create folder to store diagnostic plots in - if one does not already exist
    if not os.path.isdir(cwd + '/diagnostic_plots/' + name):
        os.mkdir(f'{cwd}/{plot_dir}{name}')

# Filter out low values Anything lower than the resolution
daily[daily<0.3] = 0

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
markov_models = fit_markov(names, occurence)
markov_model_seasonal = fit_markov(names, occurence, seasonal=True)
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
occ_corr = determine_corrmat(occ_corr, occ_corr_org,
                             markov_models=markov_model_seasonal, 
                             time_array=time_array, seasonal=True,
                             plot_error=True, plot_dir='diagnostic_plots')

# Copy rainfall correlation array - Using copy to avoid pointer issues
rainfall_corr = copy.copy(rainfall_corr_obs.values)
rainfall_corr = determine_corrmat(rainfall_corr, rainfall_corr_obs.values, corr_type='rainfall')

#%%
# =============================================================================
#  Build model for daily rainfall amounts
# =============================================================================
print('Creating model for daily rainfall amounts...')
def expon_sample(l, x):
    return (-(1/l) * np.log(1- x))

def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)

# # Build seasonal exponential model
# expon_model = np.zeros((4, df.shape[1]))
# for i in range(df.shape[1]):
#     # Extract station
#     df_temp = df[names[i]]
    
#     # Drop days with no rain
#     df_temp = df_temp.drop(df_temp[df[names[i]].values==0].index)
    
#     # Group the data by season
#     sp = df_temp.groupby([lambda x: season_dict[x.month]])
    
#     # Get the mean values
#     sp_mean = sp.mean()
#     # Extract values to the model
#     expon_model[:, i] = sp_mean.values
    
#     if diagnostic_plots:
#         ## Compare distribution with data ##
#         # Create folder to store diagnostic plots in - if one does not already exist
#         if not os.path.isdir(cwd + '/diagnostic_plots/' + names[i]):
#             plot_dir = cwd + '/diagnostic_plots/' + names[i]
#             os.mkdir(plot_dir)
        
#         # Go through each season and plot the CDF
#         for season in season_name:
#             fig, ax = plt.subplots()
            
#             # Plot the empical cdf
#             cdf(sp.get_group(season), label='Empirical')
            
#             # Draw 10.000 samples from the fitted distribution and compare
#             theoretical = np.zeros((10000,))
#             x = np.random.rand(theoretical.size)
#             l = 1 / sp_mean[season]
#             for j in range(theoretical.size):
#                 theoretical[j] = expon_sample(l, x[j])
            
#             # Plot the theoretical
#             cdf(theoretical, label='Theoretical')
            
#             # Make the plot look nice
#             ax.set_xscale('log')
#             ax.set_title(season)
#             ax.grid('Major')
#             ax.grid('Minor')
#             ax.legend()


# Build seasonal exponential model
expon_model = np.zeros((df.shape[1],))
for i in range(df.shape[1]):
    # Extract station
    df_temp = df[names[i]]
    
    # Drop days with no rain
    df_temp = df_temp.drop(df_temp[df[names[i]].values==0].index)
    
    # Extract values to the model
    expon_model[i] = df_temp.mean()
    
    if diagnostic_plots:
        fig, ax = plt.subplots()
        cdf(df_temp.values)
        theoretical = np.zeros((3000,))
        x = np.random.rand(theoretical.size)#x_array[:,i]
        l = 1 / expon_model[i]
        for j in range(theoretical.size):
            theoretical[j] = expon_sample(l, x[j])
            
        # Plot the theoretical
        cdf(theoretical, label='Theoretical')
        
        ax.set_xscale('log')

# Calculate occurence indexs for each station
print('\tcalculating occurence index..')
occ_index = calc_occindex(occurence, m, occ_corr_org)

# Initialze multip exponential model
print('\tcreating multi-exponential model...')
alpha_list = []
lambda_list = []
retbin_list = []
expon_model = np.zeros((7, m))
for i in range(occ_index.shape[1]):
    n_class = 11
    
    df_temp = pd.DataFrame(data=occ_index[:, i], columns=['occ_index'])
    df_temp = df_temp.drop(df_temp[df[names[i]].values==0].index)
    
    if i==0:
        # Categorize the data into 6(4) different classes - FIGURE OUT A WAY TO AUTO DETERMINE THE NCLASS
        _, bins = pd.qcut(df_temp['occ_index'], n_class, 
                                          duplicates='drop', retbins=True)
        
        # Save the bins
        retbin_list.append(bins)
    
    n_class = bins.size-1
    df_temp['class'] = pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(n_class), include_lowest=True)
    
    class_precip = []
    for j in range(n_class):
        id_ = df_temp.index[df_temp['class']==j].values
        class_precip.append(np.mean(df[names[i]].iloc[id_]))
        
    expon_model[:,i] = class_precip
     
#%%
# =============================================================================
# Let's create some rainfall!!
# =============================================================================    
# Initialze some stuff!
# Set random seed
np.random.seed(12)
   
## Step 1 - Create, correlated, occurence array ##    
# Generate standard normal random numbers
rnd_ = np.random.normal(0.0, 1.0, size=(m, n))

# Create correlated random, uniform, numbers
rnd = correlated_rnd(occ_corr, rnd_)

# Simulate occurences
seq = execute_markov(m, n, rnd, markov_model_seasonal, names, seasonal=True, time_array=time_array)

# Calculate occurence index
occ_index_sim = calc_occindex(seq.T, m, occ_corr)

# Create correlated random, uniform, numbers
rnd_ = np.random.normal(0.0, 1.0, size=(m, n))
x_array = correlated_rnd(rainfall_corr, rnd_).T
#x_array = np.random.rand(n,m)
## Step 3 - Determine rainfall amounts! ##    
rainfall = np.zeros((n,m))
for i in range(df.shape[1]):
    #bins = retbin_list[i]
    df_temp = pd.DataFrame(data=occ_index_sim[:, i], columns=['occ_index'])
    df_temp = df_temp.drop(df_temp[seq[i,:]==0].index)
    
    n_class = bins.size-1
    df_temp['class'] = pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(n_class), include_lowest=True)
    
    for k, seq_ in enumerate(seq[i, :]):
        if seq_==1:                                  
            # Determine rainfall amount
            x = x_array[k,i]
            class_ = df_temp['class'][k]
            temp_rain = expon_sample(1 / expon_model[class_, i], x)
            
            rainfall[k, i] = temp_rain

# # Save the simulated rainfall to a txt file, for pca.
np.savetxt('mult_gets_rainfall.dat', rainfall)
             
# =============================================================================
# Perform diagnostics of the rainfall simulator!
# =============================================================================
#%%
def mulGETS():
    ## Step 1 - Create, correlated, occurence array ##    
    # Generate standard normal random numbers
    rnd_ = np.random.normal(0.0, 1.0, size=(m, n))
    
    # Create correlated random, uniform, numbers
    rnd = correlated_rnd(occ_corr, rnd_)
    
    # Simulate occurences
    seq = execute_markov(m, n, rnd, markov_model_seasonal, names, seasonal=True, time_array=time_array)
    
    # Calculate occurence index
    occ_index_sim = calc_occindex(seq.T, m, occ_corr)
    
    # Create correlated random, uniform, numbers
    rnd_ = np.random.normal(0.0, 1.0, size=(m, n))
    x_array = correlated_rnd(rainfall_corr, rnd_).T
    ## Step 3 - Determine rainfall amounts! ##    
    rainfall = np.zeros((n,m))
    for i in range(df.shape[1]):
        #bins = retbin_list[i]
        df_temp = pd.DataFrame(data=occ_index_sim[:, i], columns=['occ_index'])
        df_temp = df_temp.drop(df_temp[seq[i,:]==0].index)
        
        n_class = bins.size-1
        df_temp['class'] = pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(n_class), include_lowest=True)
        
        for k, seq_ in enumerate(seq[i, :]):
            if seq_==1:                                  
                # Determine rainfall amount
                x = x_array[k,i]
                class_ = df_temp['class'][k]
                temp_rain = expon_sample(1 / expon_model[class_, i], x)
                
                rainfall[k, i] = temp_rain
                
    return rainfall
    
# Calculate annual statistics and lowfrequency variablitiy
def find_conf(data, conf):
    sorted_data = np.sort(data)
    lower_conf = sorted_data[int(conf/2*data.size)]
    upper_conf = sorted_data[int((1-conf/2)*data.size)]
    return (lower_conf, upper_conf)

def get_stats(df, names, i):
    df_stat = df[names[i]].divide(24).to_frame().rename(columns={names[i] : 'Int'})
    stats = rainfall_stat(df_stat, timeres=1440)
    return stats

stat_obs = {}
for i in range(names.size):
    stat_obs[names[i]] = get_stats(df, names, i)

print('Create ensemble to investigate variability of the model')
n_rlz = 10000
printProgressBar (0, n_rlz, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█')
stat_sim = {}
for k in range(n_rlz):
    rainfall = mulGETS()
    df_rainfall = pd.DataFrame(data=rainfall, index=time_array, columns=names)
    for i in range(m):
        if names[i] not in stat_sim:
            stat_sim[names[i]] = []
        stat_sim[names[i]].append(get_stats(df_rainfall, names, i))
    printProgressBar (k+1, n_rlz, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█')
    
    
for key in stat_obs:
    # Extract the observered data
    labels = ['ap', 'spwi', 'spsp', 'spsu', 'spau']
    
    # Collect the data
    gauge_ap = stat_obs[key]['ap'][0]
    gauge_spwi = stat_obs[key]['spwi'][0]
    gauge_spsp = stat_obs[key]['spsp'][0]
    gauge_spsu = stat_obs[key]['spsu'][0]
    gauge_spau = stat_obs[key]['spau'][0]
    
    gauge_data = [gauge_ap, gauge_spwi, gauge_spsp, gauge_spsu, gauge_spau]
    
    ap = []
    spwi = []
    spsp = []
    spsu = []
    spau = []
    for i in range(n_rlz):
        ap.append(stat_sim[key][i]['ap'][0])
        spwi.append(stat_sim[key][i]['spwi'][0])
        spsp.append(stat_sim[key][i]['spsp'][0])
        spsu.append(stat_sim[key][i]['spsu'][0])
        spau.append(stat_sim[key][i]['spau'][0])
        
    error_array = np.zeros((2, len(labels)))
    sst_ap   = np.median(ap)
    l, u = find_conf(np.array(ap), 0.05)
    error_array[0,0] = np.abs(l-sst_ap)
    error_array[1,0] = np.abs(u-sst_ap)
    
    sst_spwi = np.median(spwi)
    l, u = find_conf(np.array(spwi), 0.05)
    error_array[0,1] = np.abs(l-sst_spwi)
    error_array[1,1] = np.abs(u-sst_spwi)
    
    sst_spsp = np.median(spsp)
    l, u = find_conf(np.array(spsp), 0.05)
    error_array[0,2] = np.abs(l-sst_spsp)
    error_array[1,2] = np.abs(u-sst_spsp)
    
    sst_spsu = np.median(spsu)
    l, u = find_conf(np.array(spsu), 0.05)
    error_array[0,3] = np.abs(l-sst_spsu)
    error_array[1,3] = np.abs(u-sst_spsu)
    
    sst_spau = np.median(spau)
    l, u = find_conf(np.array(spau), 0.05)
    error_array[0,4] = np.abs(l-sst_spau)
    error_array[1,4] = np.abs(u-sst_spau)
    
    sst_data = [sst_ap, sst_spwi, sst_spsp, sst_spsu, sst_spau]
    
    # Plot the data
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, gauge_data, width, label='Gauge')
    rects2 = ax.bar(x + width/2, sst_data, width, label='SST', yerr=error_array)
    ax.set_ylabel('Rainfall depth [mm]')
    ax.set_title(f'Mean rainfall depth: gauge_{key} vs model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.savefig(f'{plot_dir}{key}/{key}_mean-precip.png', dpi=300)
        
    

# ## Process the simulated rainfall ##
# # Add the simulated rainfall to a dataframe
# df_rainfall = pd.DataFrame(data=rainfall, index=time_array, columns=names)

# # Calculate the correlation of the rainfall amounts
# rainfall_corr_sim = df_rainfall.corr().values

# # Calculate the simulated occurence correlation
# df_occ_sim = pd.DataFrame(data=seq.T, columns=names)  
# occ_sim_corr = df_occ_sim.corr().values

# ## Compare occurence correlation ##
# org = np.reshape(occ_corr_org, (occ_corr_org.size,))
# org[org==1] = np.nan

# sim = np.reshape(occ_sim_corr, (occ_sim_corr.size,))
# sim[sim==1] = np.nan

# fig, ax = plt.subplots()
# ax.scatter(org,sim)
# ax.set(xlim=(0.4, 1), ylim=(0.4, 1))
# diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
# ax.set(xlim=(0.4, 1), ylim=(0.4, 1))

# ax.set_xlabel('Observed correlation [-]')
# ax.set_ylabel('Simulated correlation [-]')
# ax.set_title('Occurence correlation')
# ax.grid('Major')

# # Save the figure
# fig.savefig('diagnostic_plots/occurence_correlation.png', dpi=300)

# ## Compare rainfall correlation ##
# org = np.reshape(rainfall_corr_obs.values, (rainfall_corr_obs.values.size,))
# org[org==1] = np.nan

# sim = np.reshape(rainfall_corr_sim, (rainfall_corr_sim.size,))
# sim[sim==1] = np.nan

# fig, ax = plt.subplots()
# ax.scatter(org,sim)
# ax.set(xlim=(0.4, 1), ylim=(0.4, 1))
# diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
# ax.set(xlim=(0.4, 1), ylim=(0.4, 1))

# ax.set_xlabel('Observed correlation [-]')
# ax.set_ylabel('Simulated correlation [-]')
# ax.set_title('Precip. amount correlation')
# ax.grid('Major')

# # Save the figure
# fig.savefig('diagnostic_plots/precip_correlation.png', dpi=300)




























