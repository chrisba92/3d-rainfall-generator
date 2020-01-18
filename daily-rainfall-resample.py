import pandas as pd
import numpy as np
import copy
from scipy import special
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import sys

# =============================================================================
# Functions in script
# =============================================================================
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

def determine_corrmat(corr_mat, org_corr_mat, corr_type='occurence', lr=0.1, n_sim=1000):
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
            rnd = execute_markov(m, n, rnd, markov_models, names)
                    
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
            if last_avg is not None and np.isclose(avg, last_avg, atol=1e-4) and fitness[-1]<1:
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
        
    return corr_mat

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
occ_corr = determine_corrmat(occ_corr, occ_corr_org)

# Copy rainfall correlation array - Using copy to avoid pointer issues
rainfall_corr = copy.copy(rainfall_corr_obs.values)
rainfall_corr = determine_corrmat(rainfall_corr, rainfall_corr_obs.values, corr_type='rainfall')

#%%
# =============================================================================
#  Build model for daily rainfall amounts
# =============================================================================
print('Creating model for daily rainfall amounts...')

# Calculate occurence indexs for each station
print('\tcalculating occurence index..')
occ_index = calc_occindex(occurence, m, occ_corr_org)
np.savetxt()

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
    n_class = 11
    
    df_temp = pd.DataFrame(data=occ_index[:, i], columns=['occ_index'])
    df_temp = df_temp.drop(df_temp[df[names[i]].values==0].index)
    # Categorize the data into 6(4) different classes - FIGURE OUT A WAY TO AUTO DETERMINE THE NCLASS
    if i>=0:
        _, bins = pd.qcut(df_temp['occ_index'], n_class, 
                                         duplicates='drop', retbins=True)
    n_class = bins.size-1
    df_temp['class'] = pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(n_class), include_lowest=True)
     
    # Save the class definition for the model
    retbin_list.append(bins)
        
    # Locate rainfall amount for each class and month
    month_class_mean = {}
    for k, (val, class_) in enumerate(zip(df_temp['occ_index'], df_temp['class'])):
        if class_ not in month_class_mean: # If class haven't been added to the dict, initialize it
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

    #break

from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a*x**2 + b*x + c

# Visualize the multi-expon fit
for season in season_name:
    xdata = np.arange(1, bins.size)
    ydata = []    
    for key in result:
        ydata.append(result[key][season][0])
    
    popt, _ = curve_fit(func, xdata, ydata)  
    y_plot = func(xdata, popt[0], popt[1], popt[2])
    plt.figure()
    plt.scatter(xdata, ydata)
    plt.plot(xdata, y_plot)
     
#%%
# =============================================================================
# Let's create some rainfall!!
# =============================================================================
def mulGETS(result, i):
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
    
    # Create correlated random, uniform, numbers
    x_array = correlated_rnd(rainfall_corr, rnd_).T
    
    ## Step 2 - Calculate occurence index for each station ##
    occ_index_new = calc_occindex(seq.T, m, occ_corr_org)
    
    ## Step 3 - Determine rainfall amounts! ##
    rainfall = np.zeros((n,m))
    #for i in range(occ_index_new.shape[1]):
    # Unpack model parameters
    bins   = retbin_list[i]   # Class definiton
    #result = lambda_list[i]   # Lambda values for the mutl exp function
    alpha  =  alpha_list[i]   # Alpha values for the mult exp function
    
    # Categorize the data into the 4 different class'
    df_temp = pd.DataFrame(data=occ_index_new[:, i], columns=['occ_index'])
    df_temp = df_temp.drop(df_temp[seq.T[:,i]==0].index)
    df_temp['class']= pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(bins.size-1), include_lowest=True)
    
    for k, class_ in enumerate(df_temp['class']):
        # Determine season for rainfall occurence
        season = season_dict[time_array[k].month]
        
        # Determine rainfall amount
        x = x_array[df_temp.index[k], i]
        temp_rain = []
        for j in range(bins.size-1):
            a = alpha[j]
            l = 1 / result[j][season][0]
            temp_rain.append(mult_exp(x, a, l))
        
        rainfall[df_temp.index[k], i] = np.sum(temp_rain)
            
    rainfall = pd.DataFrame(data=rainfall[:, i], index=time_array, columns=[names[i]])
    return rainfall


def get_stats(df):
    df_stat = df[names[i]].divide(24).to_frame().rename(columns={names[i] : 'Int'})
    stats = rainfall_stat(df_stat, timeres=1440)
    return stats

def convert2dict(x0, dim0, dim1):
    temp = np.reshape(x0, (dim0, dim1))
    result = {}
    for i in range(dim0):
        result[i] = {}
        for j in range(dim1):
            result[i][season_name[j]] = [temp[i,j]]

    return result

def fun(x, *args):
    result = convert2dict(x, args[0], args[1])
    rainfall = mulGETS(result, args[3])
    ap_sim = get_stats(rainfall)['ap'][0]
    score = (np.abs(args[2] - ap_sim)) / args[2]
    return score

# from scipy.optimize import minimize, differential_evolution
# for i in range(df.shape[1]):
#     # Get target value for the optimizer
#     ap_obs = get_stats(df)['ap'][0]
    
#     # Extract lambda values to get inital guess of the optimizer
#     x0 = np.zeros((4 * len(lambda_list[i]),))
    
#     bounds = []
#     for _ in range(4 * len(lambda_list[i])):
#         bounds.append((0.01,10))
    
#     j = 0
#     for class_ in lambda_list[i]:
#         for season in lambda_list[i][class_]: 
#             x0[j] = lambda_list[i][class_][season][0]
#             j+=1
        
#     #result_temp = convert2dict(x0, len(lambda_list[i]), 4)
#     #score = fun(x0, (len(lambda_list[i]), 4, ap_obs, i))
#     res = differential_evolution(fun, bounds=bounds, args=(len(lambda_list[i]), 4, ap_obs, i), disp=True, workers=3)
#     break
    
# # Initialze some stuff!
# # Set random seed
# np.random.seed(1234)
    
# # Total number of sequences to model
# n = occurence.shape[0]

# # Set number of stations
# m = occurence.shape[1]

# # Create time array
# time_array = pd.date_range(start=df.index[0], periods=n)

# season_dict = {12 : 'DJF', 1 : 'DJF', 2 : 'DJF',
#                3 : 'MAM', 4 : 'MAM', 5 : 'MAM',
#                6: 'JJA', 7: 'JJA', 8: 'JJA',
#                9: 'SON', 10: 'SON',  11: 'SON'}
   
# ## Step 1 - Create, correlated, occurence array ##    
# # Generate standard normal random numbers
# rnd_ = np.random.normal(0.0, 1.0, size=(m, n))

# # Create correlated random, uniform, numbers
# rnd = correlated_rnd(occ_corr, rnd_)

# # Simulate occurences
# seq = execute_markov(m, n, rnd, markov_models, names)

# # Create correlated random, uniform, numbers
# x_array = correlated_rnd(rainfall_corr, rnd_).T

# ## Step 2 - Calculate occurence index for each station ##
# occ_index_new = calc_occindex(seq.T, m, occ_corr_org)

# ## Step 3 - Determine rainfall amounts! ##
# rainfall = np.zeros((n,m))
# for i in range(occ_index_new.shape[1]):
#     # Unpack model parameters
#     bins   = retbin_list[i]   # Class definiton
#     result = lambda_list[i]   # Lambda values for the mutl exp function
#     alpha  =  alpha_list[i]   # Alpha values for the mult exp function
    
#     # Categorize the data into the 4 different class'
#     df_temp = pd.DataFrame(data=occ_index_new[:, i], columns=['occ_index'])
#     df_temp = df_temp.drop(df_temp[seq.T[:,i]==0].index)
#     df_temp['class']= pd.cut(df_temp['occ_index'], bins=bins, labels=np.arange(bins.size-1), include_lowest=True)
    
#     for k, class_ in enumerate(df_temp['class']):
#         # Determine season for rainfall occurence
#         season = season_dict[time_array[k].month]
        
#         # Determine rainfall amount
#         x = x_array[df_temp.index[k], i]
#         temp_rain = []
#         for j in range(bins.size-1):
#             a = alpha[j]
#             l = 1 / result[j][season][0]
#             temp_rain.append(mult_exp(x, a, l))
        
#         rainfall[df_temp.index[k], i] = np.sum(temp_rain)

# # Save the simulated rainfall to a txt file, for pca.
# np.savetxt('mult_gets_rainfall.dat', rainfall)
             
# # =============================================================================
# # Perform diagnostics of the rainfall simulator!
# # =============================================================================
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


# # Calculate annual statistics and lowfrequency variablitiy
# def get_stats(df):
#     df_stat = df[names[i]].divide(24).to_frame().rename(columns={names[i] : 'Int'})
#     stats = rainfall_stat(df_stat, timeres=1440)
#     return stats

# stat_obs = {}
# for i in range(names.size):
#     stat_obs[names[i]] = get_stats(df)

# stat_sim = {}
# for _ in range(1):
#     df_mulgets = pd.DataFrame(data=mulGETS(), index=time_array, columns=names)    
#     for i in range(1):
#         if names[i] not in stat_sim:
#             stat_sim[names[i]] = []
            
#         stat_sim[names[i]].append(get_stats(df_mulgets))
        
    






























