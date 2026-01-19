# ---------------------------------------------------------------------------------------------------------------------------------------
#  --------------------------------------------------------- REQUIRED PACKAGES  ---------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import ipywidgets as widgets

from pathlib import Path
from tqdm import tqdm
import warnings


# ---------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- FUNCTIONS FROM STRAIN SYNCHRONIZATION NOTEBOOK --------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

# Define import function
def read_meta(path, experiments):
    """
    Imports and processes experimental metadata from an Excel spreadsheet.

    This function reads data from an Excel file at the specified path, extracting 
    and organizing cross-sectional and load data for a list of specified experiments.
    It processes the 'Masses_Samples' sheet to obtain cross-sectional data, and 
    matches experiment names with corresponding sheets to extract load data. The 
    function calculates the stress for each sample based on the cumulative load and 
    sample area, then compiles the processed data into a DataFrame.

    Parameters:
    -----------
    path : str
        The file path to the Excel spreadsheet containing the experimental data.
    experiments : list of str
        The list of experiments to extract data for, matched against the sheet names.

    Returns:
    --------
    df_meta : pandas.DataFrame
        A DataFrame containing the compiled metadata for the specified experiments,
        with multi-level indices of 'experiment' and 'sample', and columns for 
        'name', 'weights', 'area', and 'stress'.

    Raises:
    -------
    Prints an error message if multiple or no matching sheets are found for a given 
    experiment name.
    """
    # Import spreadsheet
    dfs_loads = pd.read_excel( path, sheet_name=None )
        
    sheets = list( dfs_loads.keys() )
    
    # Extract cross-sectional data from spreadsheet
    df_areas = dfs_loads['Masses_Samples'].copy()
    df_areas['Experiment'] = df_areas['Experiment'].ffill()
    df_areas.rename(columns={ df_areas.columns[0]: 'Properties' }, inplace = True)
    df_areas.set_index(['Experiment', 'Properties'], inplace=True)
    
    
    # Extract load data from spreadsheet
    df_loads = dict()
    for experiment in experiments:
        
        # Match correct sheet
        sheet_idx = np.nonzero( [ experiment.lower() in sheet.lower() or sheet.lower() in experiment.lower() for sheet in sheets ] )[0]
        if len(sheet_idx) != 1:
            print(f'\033[1m\033[91mError:\033[0m {len(sheet_idx):d} matches between spreadsheet names and experiment {experiment:s} found.')
            continue # Breakout criterion
            
        # Extract and arrange sheet data
        sheet = dfs_loads[ sheets[sheet_idx[0]] ]
        col_old, col_new = sheet.columns, ['topic', 'object', 'property']
        sheet.rename( columns={col_old[0]:col_new[0], col_old[1]:col_new[1], col_old[2]:col_new[2]}, inplace=True )
        sheet[col_new] = sheet[col_new].ffill(); sheet.fillna( value=0, inplace=True )
        sheet.set_index( col_new, inplace=True )
        sheet.drop( columns=[ col for col in col_old[3:] if 'unnamed' in str(col).lower() ], inplace=True )
        
        df_loads[experiment] = sheet
    
    
    # Iterate through all experiments
    dfs_meta = []
    for experiment in experiments:
        
        # Identify sample numbers
        samples = [column.upper().replace(' ', '_') for column in df_areas.columns.get_level_values(0) if 'SAMPLE' in column.upper()]
        sample_idx = [ int(sample.split('_')[-1]) for sample in samples ]
    
        # Extract cross section
        df_area = df_areas.rename( columns=lambda s: s.upper().replace(' ', '_'), index=lambda s: 'MACRO_' + s.upper(), level=0 )
        areas = [df_area.loc[(experiment, 'Thickness [mm]'), sample] * df_area.loc[(experiment, 'Width [mm]'), sample] for sample in samples ]
        
        # Extract load data
        df_loads[experiment].sort_index( inplace=True )
        topic, key, prop = 'Load calibration', 'Weight ', 'cumulative load [N]'
        weight_idx = [ load for load in df_loads[experiment].loc[topic].index.get_level_values(0).unique() if key in load ]
        weights = [ [ i for i in df_loads[experiment][idx].loc[topic,weight_idx,prop] if i != 0 ] for idx in sample_idx ]
        names = [ df_loads[experiment][idx].loc['Sample','Name'].iloc[0] for idx in sample_idx ]
    
        stresses = [w[-1]/a for a, w in zip(areas, weights) ]
        
        df = pd.DataFrame( {'experiment': experiment, 'sample': samples, 'name': names, 'weights': weights, 'area': areas, 'stress': stresses} )
        dfs_meta.append(df)
    
    # Assemble table
    df_meta = pd.concat( dfs_meta )
    df_meta.set_index( ['experiment', 'sample'], inplace=True )
    df_meta.sort_index( inplace=True )
    
    return df_meta


# Define strain synchronization function
def sync_cameras(df, shear=False):
    '''
    Synchronizes the strain data of the left and right camera. The returned
    times are the average time of two adjacent time steps.
    Input:
        df      (pd.DataFrame) : DataFrame with index columns 'side' and 'times'
                                 and columns with strain data.
        shear   (bool)         : If True, the strains get multiplied with their
                                 average sign to make both camera's signs equal.
    Output:
        df_sync (pd.DataFrame) : DataFrame where the returned columns are the average
                                 columns over 'left' and 'right' side.
    '''
    # Initialize
    df = df.copy()
    sides = ['left', 'right']
    
    # Unify sign if field is shear strain
    if shear:
        for side in sides:
            df.loc[side] = (df.loc[side] * np.sign(df.loc[side].mean())).to_numpy()
    
    # Identify common times
    dtol = 25.0 # maximum time difference in seconds where left and right strain get combined
    dtimes_sides = [(df.loc[side].index - df.loc[side].index[0]).total_seconds() for side in sides]
    dtimes = np.sort(np.concatenate(dtimes_sides))
    d = np.where(np.diff(dtimes) < dtol)[0]
    times_avg = (dtimes[d] + dtimes[d+1]) / 2.0
    
    # Construct synchronized dataframe
    date_0 = df.loc['left'].index[0] + (df.loc['left'].index[1] - df.loc['right'].index[0]) / 2.0
    date_avg = date_0 + pd.to_timedelta( times_avg, unit='s')
    df_sync = pd.DataFrame(index=date_avg)
    for column in df.columns:
        times_left, times_right = dtimes_sides[0], dtimes_sides[1]
        col_left = sp.interpolate.interp1d(times_left, df.loc['left', column], kind='nearest', fill_value='extrapolate')(times_avg)
        col_right = sp.interpolate.interp1d(times_right, df.loc['right', column], kind='nearest', fill_value='extrapolate')(times_avg)
        df_sync[column] = (col_left + col_right) / 2.0

    # Return results
    return df_sync

# sync_cameras(df_strains.loc[ ('MACRO_VISCO_RH65_02', 'SAMPLE_2', slice(None), 'plot_eyy_formatted') ])


# Define stress synchronization function
def sync_loads(df, df_meta):
    '''
    Add the stresses and loading states as new columns to the given DataFrame.
    Input:
        df      (pd.DataFrame) : DataFrame with strain data in columns.
        df_meta (bool)         : DataFrame with index columns 'experiment' and 'sample'
                                 and corresponding 'weights' and 'area'.
    Output:
        df      (pd.DataFrame) : DataFrame with added stress and loading state columns.
    '''
    # Initialize
    df = df.copy()
    loads = df_meta.loc[(experiment, sample), 'weights']
    area = df_meta.loc[(experiment, sample), 'area']
    
    # Identify rows with load changes
    a = df.index.diff() < pd.Timedelta('5min')
    idx = np.where(a)[0]-1
    a[idx] = np.ones(len(idx), dtype=bool)
    
    # Find connected load changes
    idx = np.where(a)[0]
    b = np.where(np.diff(idx) > 1)[0]
    if len(b) != 1:
        raise ValueError(f'Invalid number of connected loading groups found. Expected 1 group, but found {len(b):d}.')
    group_loading = idx[:b[0]+1]
    group_unloading = idx[b[0]+1:]
    
    # Add loading and unloading
    stress = pd.Series( data=np.full(len(a), np.nan) )
    stress[group_loading[:len(loads)+1]] = np.array([0] + loads) / area
    stress[group_unloading[:len(loads)+1]] = np.flip([0] + loads) / area
    
    # Fill gaps
    stress = stress.ffill().bfill()
    
    # Create state column
    state = np.full(len(a), 'unloaded', dtype='<U10')
    state[group_loading] = 'loading'
    state[group_unloading] = 'unloading'
    state[np.max(group_loading)+1 : np.min(group_unloading)] = 'loaded'
    
    df['stress'] = stress.to_numpy()
    df['state'] = state

    return df


# ---------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- FUNCTIONS FROM VISCOELASTICITY FITTING NOTEBOOK -------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

# Interpolate elasticity
def interpolate_elasticity(df_elasticity, batch_rh):
    """
    Interpolates elasticity values in `df_elasticity` for a given relative humidity (RH) `batch_rh`
    if it is not already present. Uses surrounding RH values to estimate the elasticity at `batch_rh`
    by performing a linear interpolation based on moisture contents derived from adsorption and desorption.

    Parameters:
    df_elasticity (pd.DataFrame): DataFrame containing elasticity values indexed by relative humidities.
    batch_rh (float): The target relative humidity for which elasticity needs to be interpolated.

    Returns:
    pd.DataFrame: Updated DataFrame with interpolated elasticity at `batch_rh` added.
    """
    rhs = df_elasticity.index.unique()
    if not batch_rh in rhs:
        # Get next relative humidities
        rh0 = rhs[rhs < batch_rh].max()
        rh1 = rhs[rhs > batch_rh].min()
        rh_target = batch_rh
    
        # Get moisture contents
        mc0 = (adsorption_rh2mc(rh0/100) + desorption_rh2mc(rh0/100)) / 2
        mc1 = (adsorption_rh2mc(rh1/100) + desorption_rh2mc(rh1/100)) / 2
        mc_target = (adsorption_rh2mc(rh_target/100) + desorption_rh2mc(rh_target/100)) / 2
    
        # Interpolate between moisture contents
        a = (mc_target - mc0) / (mc1 - mc0)
        elasticity_target = df_elasticity.loc[rh0] + a*(df_elasticity.loc[rh1] - df_elasticity.loc[rh0])
        df_elasticity = pd.concat( [df_elasticity, pd.DataFrame(data=elasticity_target, columns=[rh_target]).T] ).sort_index()

    return df_elasticity


# Define function to get elastic stiffness
def get_stiffness(df_elasticity, sample_name, component, rh):
    '''
    Select the correct stiffness for a sample.
    Input:
        df_elasticity (pd.DataFrame) : DataFrame with elasticity constants, indexed by relative humidity.
        sample_name   (str)          : Name of the sample, indicating the sample.
    Output:
        stiffness     (float)        : Selected stiffness value.
    '''
    # Initialize
    ser_params = df_elasticity.loc[rh]
    
    sample_type = sample_name.split('_')[1].split('-')[0]
    loading_type = sample_type[0].lower()
    direction_type = sample_type[1:].upper()
    lateral = {'T': 'R', 'R': 'L', 'L': 'R', 'LX': 'T'}
    direction = direction_type.strip('X')

    # Check input
    if loading_type == 's' and component != 'exy':
        raise ValueError(f'Invalid component "{component}" for shear sample type.')
    elif loading_type != 's' and component == 'exy':
        raise ValueError(f'Invalid component "{component}" for uniaxial sample type.')

    # Identify stiffness parameter
    match component:
        case 'exx':
            stiffness = -ser_params[f'E_{direction}'] / ser_params[f'nu_{direction}{lateral[direction_type]}']
        case 'exy':
            try:
                stiffness = ser_params[f'G_{direction}']
            except:
                stiffness = ser_params[f'G_{direction[::-1]}']
        case 'eyy':
            stiffness = ser_params[f'E_{direction}']
        case _:
            raise ValueError(f'Invalid component "{component}".')
            
    return stiffness


# -----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- FUNCTIONS FROM CLIMATES COMPARISON NOTEBOOK -------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# Define boxplot function
def boxplot(df, x, y, hue, order=None, legend=True, ax=None, fc='blue', ec='black', marker='o',
            box_width=0.2, lw=1.2, alpha=1.0, offset_fun=None):
    # Initialize plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Set box settings
    boxprops = [
        dict(facecolor=fc, edgecolor=ec, alpha=alpha),
        dict(facecolor=fc, edgecolor=ec, alpha=alpha),
        dict(facecolor=fc, edgecolor=ec, alpha=alpha)
    ]
    whiskerprops = [
        dict(color=ec, alpha=alpha),
        dict(color=ec, alpha=alpha),
        dict(color=ec, alpha=alpha)
    ]
    capprops = [
        dict(color=ec, alpha=alpha),
        dict(color=ec, alpha=alpha),
        dict(color=ec, alpha=alpha)
    ]
    medianprops = [
        dict(color=ec, linewidth=lw, alpha=alpha),
        dict(color=ec, linewidth=lw, alpha=alpha),
        dict(color=ec, linewidth=lw, alpha=alpha)
    ]
    flierprops = [
        dict(marker=marker, markeredgecolor=ec, alpha=alpha),
        dict(marker=marker, markeredgecolor=ec, alpha=alpha),
        dict(marker=marker, markeredgecolor=ec, alpha=alpha)
    ]
    
    # Initialize categories
    categories = df[x].unique() if order is None else order
    category_positions, i = dict(), 0
    for c in categories:
        i += 1
        category_positions[c] = i

    # Initialize groups
    groups = df[hue].unique()
    if offset_fun is None:
        gmin, gmax = np.min(groups), np.max(groups)
        offset_fun = lambda g: 0.5 * ((g - gmin) / (gmax - gmin) - 0.5)
    group_offsets = dict()
    for g in groups:
        group_offsets[g] = offset_fun(g)

    # Collect data
    for category in categories:
        for i, group in enumerate(groups):
            # Select subset
            data = df[(df[x] == category) & (df[hue] == group)][y]

            # Store data
            values = data.dropna().values
            positions = category_positions[category] + group_offsets[group]

            # Create boxplot
            bp = ax.boxplot(
                values,
                positions=[positions],
                widths=box_width,
                patch_artist=True,
                medianprops=medianprops[i],
                showfliers=True,
                boxprops=boxprops[i],
                whiskerprops=whiskerprops[i],
                flierprops=flierprops[i],
                capprops=capprops[i]
            )

    # Setup x ticks
    ax.set_xticks(list(category_positions.values()))
    ax.set_xticklabels(list(category_positions.keys()))

    # Add legend
    if legend:
        elements = []
        for i, g in enumerate(groups):
            elements.append( mpl.patches.Patch(label=g, **boxprops[i]) )
        ax.legend(handles=elements, title='Group')
        
    return fig, ax


# -----------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- FUNCTIONS FROM CREEP RESULTS NOTEBOOK -------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# Import moisture contents
def import_creep_moisture(file_meta, samples=None):
    '''
    Imports and processes moisture content data from a specified Excel file.

    This function reads a spreadsheet containing moisture content data, cleans the data by removing 
    unnecessary columns and replacing invalid entries, calculates mean and standard deviation values 
    for each experimental condition, and fills missing data using group averages based on relative humidity (RH).
    It also associates sample names with calculated moisture content values and computes summary statistics 
    (mean, standard deviation, minimum, and maximum) for moisture content grouped by RH.

    Parameters:
    file_meta (str): The file path to the Excel spreadsheet containing the data.
    samples (list): Samples to include in the averaging. Average over all samples if None (default).

    Returns:
    tuple: A tuple containing:
        - df_mc_samples (DataFrame): A DataFrame linking sample names to their moisture content means, 
          standard deviations, and RH values.
        - mc_avg (DataFrame): A DataFrame containing summary statistics (mean, std, min, max) of 
          moisture content for each RH group.
    '''
    # Read spreadsheet
    df_mc = pd.read_excel(file_meta, sheet_name='Masses_Twins', index_col=[0,1])
    
    # Cleanup table
    df_mc.dropna(how='all', inplace=True)
    df_mc.drop(columns=['Note'], inplace=True)
    df_mc.replace('#VALUE!', np.nan, inplace=True)
    df_mc.sort_index(inplace=True)
    df_mc = df_mc.loc['Moisture content [-]']
    
    # Add additional information
    df_mc['mean'], df_mc['std'] = df_mc.mean(axis=1), df_mc.std(axis=1)  # do not set subsequentially for noting including mean in the std calculation
    df_mc['rh'] = [int(experiment.split('_')[1].strip('RH')) for experiment in df_mc.index]
    
    # Fill missing values
    df_mc_mean = df_mc[['mean', 'rh']].groupby('rh').mean()
    df_mc_std = df_mc[['mean', 'rh']].groupby('rh').std()
    na_idx = df_mc['mean'].isna()
    df_mc.loc[na_idx, 'mean'] = df_mc_mean.loc[df_mc.loc[na_idx, 'rh'], 'mean'].to_numpy()
    df_mc.loc[na_idx, 'std'] = df_mc_std.loc[df_mc.loc[na_idx, 'rh'], 'mean'].to_numpy()
    
    # Import sample names
    df_meta = read_meta(file_meta, ['MACRO_' + name for name in df_mc.index.str.upper()])
    
    # Link sample names to moisture content
    df_mc_samples = pd.DataFrame(data={'sample': df_meta['name'], 'experiment': df_meta.index.get_level_values('experiment').str.strip('MACRO_')})
    df_mc_samples = pd.DataFrame(data={'sample': df_meta['name'].to_numpy(), 'experiment': df_meta.index.get_level_values('experiment').str.strip('MACRO_').str.replace('VISCO', 'Visco')})
    df_mc_samples['mc_mean'] = df_mc.loc[df_mc_samples['experiment'], 'mean'].to_numpy()
    df_mc_samples['mc_std'] = df_mc.loc[df_mc_samples['experiment'], 'std'].to_numpy()
    df_mc_samples['rh'] = df_mc.loc[df_mc_samples['experiment'], 'rh'].to_numpy()
    df_mc_samples.set_index('sample', inplace=True)
    
    # Calculate mean moisture content of valid samples
    if samples is None:
        samples = df_mc_samples.index.get_level_values('sample').unique()
    mc_avg = df_mc_samples.loc[ samples, ['mc_mean', 'rh'] ].groupby('rh')['mc_mean'].agg(['mean', 'std', 'min', 'max']) * 100

    return df_mc_samples, mc_avg


# ---------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- FUNCTIONS FROM MECHANOSORPTIVE FITTING NOTEBOOK -------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

# Define RH to moisture content relation
adsorption_rh2mc = lambda phi: phi / ( 3.308 + 12.087*phi - 11.839*phi**2 )
desorption_rh2mc = lambda phi: phi / ( 2.393 + 10.122*phi -  8.935*phi**2 )

adsorption_mc2rh = lambda mc: ( np.sqrt( 4*3.308*11.839*mc**2 + 12.087**2*mc**2 - 2*12.087*mc + 1 ) + 12.087*mc - 1.0 ) / (2 * 11.839*mc)
desorption_mc2rh = lambda mc: ( np.sqrt( 4*2.393* 8.935*mc**2 + 10.122**2*mc**2 - 2*10.122*mc + 1 ) + 10.122*mc - 1.0 ) / (2 *  8.935*mc)


# Transform relative humidity to moisture content
def calculate_mc(rh):
    """
    Calculate the moisture content (MC) from relative humidity (RH) values using sorption curves.

    This function computes the moisture content of a material based on its relative humidity over time, using
    predefined sorption and desorption curves. The initial condition assumes a desorption curve, but the function
    adapts dynamically to changes in humidity, switching between adsorption and desorption as needed.

    Parameters:
    -----------
    rh : pandas.Series
        A time series of relative humidity values as a percentage (0 to 100).

    Returns:
    --------
    numpy.ndarray
        An array of moisture content values corresponding to each relative humidity input value.

    Notes:
    ------
    - The function starts with the assumption that the sorption process begins with desorption.
    - If the relative humidity exceeds the right boundary, the function switches to adsorption.
    - If the relative humidity goes below the left boundary, the function switches to desorption.
    - Between the boundaries, the moisture content remains unchanged.

    Dependencies:
    -------------
    - The function relies on external functions `adsorption_rh2mc` and `desorp_mc2rh` to convert between 
      relative humidity and moisture content for adsorption and desorption, respectively.
    - The function uses NumPy for array operations.

    Example:
    --------
    >>> import pandas as pd
    >>> rh_series = pd.Series([50, 60, 55, 40, 45])
    >>> mc_values = calculate_mc(rh_series)
    >>> print(mc_values)
    array([...])  # Example output
    """
    # Start settings
    sorption_curve = 'desorption'
    
    # Initialize values
    rh = rh / 100.
    rh_0 = rh.iloc[0]
    mc = np.zeros( len(rh) )
    if sorption_curve == 'adsorption':
        mc[0] = adsorption_rh2mc(rh_0)
        rh_left = desorp_mc2rh(mc[0])
        rh_right = rh_0
    else:
        mc[0] = desorp_rh2mc(rh_0)
        rh_left = rh_0
        rh_right = adsorption_mc2rh(mc[0])
    
    # Select sorption curve
    for i in range(1, len(rh)):
        rh_i = rh.iloc[i]
        if rh_i > rh_right:
            # Adsorption
            mc[i] = adsorption_rh2mc(rh_i)
            rh_left = desorp_mc2rh(mc[i])
            rh_right = rh_i
        elif rh_i < rh_left:
            # Desorption
            mc[i] = desorp_rh2mc(rh_i)
            rh_left = rh_i
            rh_right = adsorption_mc2rh(mc[i])
        else:
            # Between sorption and desorption
            mc[i] = mc[i-1]

    return mc


# Define mechanosorptive model
def calculate_ms_model(times, moisture, stress, C_inv, mu, gamma, eps_ms, progress_bar=True):
    '''
    Calculates the mechanosorptive strains at n datapoints of a KV series with m elements.
    Input:
        times    (nx1 ndarray) : Evaluated time points (must respect a sufficiently small dt and already sorted)
        moisture (nx1 ndarray) : Moisture content matching to t
        stress   (nx1 ndarray) : Stress matching to t
        C_inv    (nx1 ndarray) : Elastic compliance C_inv(t) (inverse Young's modulus) matching to t
        mu       (mx1 ndarray) : KV elements' characteristic moisture changes
        gamma    (mx1 ndarray) : KV elements' characteristic compliances
        eps_ve   (mx1 ndarray) : KV elements' starting strains; the variable is permutable
                                 and will be updated with the KV elements' new strains
    Output:
        strain_ms (nx1 ndarray) : Mechanosorptive strain at each time point
    '''
    # Increment over moisture
    strain_ms = np.zeros(len(moisture))
    for i in tqdm( np.arange(len(moisture)-1), disable=not progress_bar):
        # Calculate strain
        strain_ms[i] = np.sum(eps_ms)

        # Calculate strain increment
        domega = np.abs( moisture[i+1]-moisture[i] ) # / (times[i+1]-times[i])
        deps_ms = domega / mu * ( C_inv[i] / gamma * stress[i] - eps_ms )
        eps_ms += deps_ms # * (times[i+1]-times[i])

    # Calculate last step
    strain_ms[-1] = np.sum(eps_ms)

    return strain_ms


# Function definitions for refining model input
def refine_times_by_dt(times, dt):
    '''
    Refines the times such that the time steps are approximately dt.
    Input:
        times     (np.array) : Times to be refined
        dt        (float)    : New time step size
    Output:
        new_times (np.array) : Refined times array
        index     (np.array) : Indexes in new_times that match the values in times
    '''
    # Get refined sections
    time_sections = [ np.arange(times[i], times[i+1], dt) for i in range(len(times)-1) ]
    time_sections += [[times[-1]]] # add last time

    # Get new times
    new_times = np.concatenate(time_sections)

    # Get indexes matching values of input array
    n = [len(t) for t in time_sections] # length per section
    index = np.concatenate([[0], np.cumsum(n)[:-1]])

    return new_times, index