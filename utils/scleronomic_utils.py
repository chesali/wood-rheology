"""
Module for Data Processing and Visualization in Material Science Experiments

This module provides a collection of functions designed for processing, visualizing, and exporting data 
from material science experiments. The functions are organized into several categories based on their 
purpose: preprocessing, creep fitting, processing, and elasticity calculations.

Key Features:
-------------
- **Preprocessing Functions**: 
  - `get_json_file(batch_name)`: Converts batch names to corresponding JSON filenames.
  - `visualize_strains(...)`: Visualizes strain data for a given sample, optionally overlaying additional 
    data from a Zwick source.
  - `find_cycles(...)`: Identifies and classifies cycles in data based on peak information.

- **Creep Fitting Functions**:
  - `find_plateaus(...)`: Extracts force plateaus from data, categorizing them into loading, creep, 
    unloading, and relax intervals.
  - `valid_plateaus(df_plateaus)`: Filters plateaus to ensure only complete cycles are retained.

- **Processing Functions**:
  - `find_linear_increments(...)`: Calculates linear increments between successive cycles in linear 
    fit parameters.
  - `df_find_linear_increments(df_elasticity)`: Processes multi-indexed DataFrame to calculate linear 
    increments for each sample.

- **Utility Functions**:
  - `import_xlsx(file, key)`: Imports and concatenates data from all sheets in an Excel file with 
    a specific keyword, assuming a uniform structure across sheets.
  - `export_df_to_xslx(...)`: Exports a DataFrame to an Excel file, replacing or appending sheets as necessary.
  - `insert_into_df(original_df, fill_df)`: Overwrites values in one DataFrame with another based on 
    common indices.

Dependencies:
-------------
- numpy
- pandas
- scipy
- matplotlib
- ipywidgets
- tqdm
- pathlib

This module is structured to support efficient data handling in experimental workflows, allowing users 
to preprocess data, fit model parameters, visualize results, and export findings for further analysis.

"""

# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------- REQUIRED PACKAGES  ---------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scipy as sp
import lmfit as lf

import matplotlib.pyplot as plt
import ipywidgets as widgets
from tqdm.notebook import tqdm

from pathlib import Path


# ---------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ FUNCTIONS FROM PREPROCESSING NOTEBOOK ------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

def get_json_file(batch_name):
    abbr = batch_name.strip('data_')
    if '_C' in batch_name.upper():
        json_name = f"Druck_Zyklisch_{abbr:s}.json"
    elif '_T' in batch_name.upper():
        json_name = f"Zug_Zyklisch_{abbr:s}.json"
    elif '_S' in batch_name.upper():
        json_name = f"Schub_Zyklisch_{abbr:s}.json"
    else:
        raise ValueError(f'Cannot transform batch name {batch_name} to JSON file name.')
    return json_name


def sync_time(x0, y0, x1, y1):
    # Synchronize x-scale by interpolation
    f0 = sp.interpolate.interp1d(x0, y0)
    f1 = sp.interpolate.interp1d(x1, y1)

    x_min = np.max([ np.min(x0), np.min(x1) ])
    x_max = np.min([ np.max(x0), np.max(x1) ])
    x = np.linspace(x_min, x_max, 10000)
    a, b = f0(x), f1(x)

    # Correlate signals with FFT
    af = sp.fft.fft(a)
    bf = sp.fft.fft(b)
    c = sp.fft.ifft(af * np.conj(bf))
    
    shift_index = np.argmax(abs(c)) # shift by index
    shift_x = shift_index * np.median(np.diff(x)) # shift of x1 by x-value

    # Return results
    return shift_x


def visualize_strains(df, sample, xcolumn, ycolumn, ax, df_zwick=None):
    """
    Visualize strain data for a given sample.

    This function plots strain data from a given DataFrame on the specified axes. It supports plotting
    data from multiple cameras and optionally overlays additional strain data from a Zwick source 
    if provided.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing strain data indexed by sample and camera. It should have columns
        corresponding to the `xcolumn` and `ycolumn`.

    sample : str
        The identifier for the sample to be visualized. The function will plot data specifically
        for this sample.

    xcolumn : str
        The column name in `df` and `df_zwick` representing the x-axis data to be plotted.

    ycolumn : str
        The column name in `df` and `df_zwick` representing the y-axis data to be plotted.

    ax : matplotlib.axes.Axes
        The matplotlib Axes object on which the data will be plotted.

    df_zwick : pandas.DataFrame, optional
        An optional DataFrame containing additional strain data from a Zwick source. If provided,
        it should be indexed by sample with columns for `width` and `thickness`, and optionally
        contain `xcolumn` and `ycolumn` data for plotting.

    Notes:
    ------
    - The function clears the current contents of `ax` before plotting.
    - Each unique camera in `df` is plotted as a separate line on the graph.
    - If `df_zwick` is provided and contains the necessary data, it is overlaid on the plot and
      the title is updated to include sample dimensions from `df_zwick`.
    - The function automatically adds grid lines, axis labels, a legend, and a title to the plot.
    """
    # Initialize
    ax.cla()
    
    # Plot the strains
    df = df.loc[sample]
    for cam in df.index.unique():
        ax.plot(df.loc[cam, xcolumn], df.loc[cam, ycolumn], label=cam)
    
    ax.set_xlabel(xcolumn)
    ax.set_ylabel(ycolumn)
    ax.grid(True)

    # Plot the Zwick data
    if not df_zwick is None:
        df_zwick = df_zwick.loc[sample]
        title = f'{sample:s}, w={df_zwick["width"]}, t={df_zwick["thickness"]}'
        if xcolumn in df_zwick.index and ycolumn in df_zwick.index:
            ax.plot(df_zwick.loc[xcolumn], df_zwick.loc[ycolumn], label='zwick')
    else:
        title=sample
        
    ax.set_title(title)
    ax.legend()


def find_cycles(x, y, df_peaks, exclude=[]):
    """
    Identify and classify cycles in a given dataset based on peak information.

    This function analyzes sequences of data points, identifying cycles between specified peaks.
    Each cycle is classified as either 'loading', 'unloading', or 'invalid' based on the trend of the
    y-values from start to end. Cycles are indexed and stored in a DataFrame.

    Parameters:
    -----------
    x : array-like
        The x-coordinates of the data points.

    y : array-like
        The y-coordinates of the data points, used to determine the cycle status.

    df_peaks : pandas.DataFrame
        A DataFrame containing information about the peaks. It must include columns 'index', 'x',
        and 'y', representing the indices, x-coordinates, and y-coordinates of peaks, respectively.

    exclude : list, optional
        A list of indices to exclude from the cycle data. This allows for selective removal of
        certain data points from cycles.

    Returns:
    --------
    pandas.DataFrame
        A concatenated DataFrame indexed by 'cycle' and 'status', containing columns:
        - 'index': The indices of the data points within each cycle.
        - 'x': The x-coordinates of the data points within each cycle.
        - 'y': The y-coordinates of the data points within each cycle.
        - 'cycle': The cycle number, starting from 0 for the first loading cycle.
        - 'status': The cycle status, either 'loading', 'unloading', or 'invalid'.

    Notes:
    ------
    - A cycle is classified as 'loading' if the y-value increases from start to end, and 'unloading'
      if it decreases. If the start and end y-values are equal, the cycle is marked as 'invalid'.
    - A warning is printed if a cycle is marked as 'invalid' due to equal start and end y-values.
    - The function uses zero-based indexing for cycle numbers.
    """
    # Extract peak information
    peaks_idx = df_peaks['index'] # point indexes of peaks
    peaks_x = df_peaks['x']
    peaks_y = df_peaks['y']

    # Iterate over all peaks
    cycle_end_last = 0
    cycle_i = -1
    dfs_cycles = []
    for i in range(len(peaks_idx)):
        # Extract cycles coordinates
        cycle_start, cycle_end = cycle_end_last, peaks_idx.iloc[i]
        cycle_idx = np.delete(np.arange(cycle_start, cycle_end+1), exclude)
        cycle_x = x[cycle_idx]
        cycle_y = y[cycle_idx]

        # Determine status
        if cycle_y[0] < cycle_y[-1]:
            status = 'loading'
            cycle_i += 1
        elif cycle_y[0] > cycle_y[-1]:
            status = 'unloading'
        else:
            print(f'\033[91mWarning:\033[00m Insufficient data points. Forces for cycle {cycle_i:d} at start ({cycle_y[0]:.3f}) and end ({cycle_y[-1]:.3f}) are equal.')
            status = 'invalid'

        # Update variables
        dfs_cycles.append( pd.DataFrame( data={'cycle': cycle_i, 'status': status, 'index': cycle_idx, 'x': cycle_x, 'y': cycle_y} ) )
        cycle_end_last = cycle_end

    # Return results
    return pd.concat(dfs_cycles).set_index(['cycle', 'status'])


def visualize_cycles(df_cycles, cycle_ids, ax):
    """
    Plot cycles from a DataFrame onto a specified Axes object.

    This function visualizes multiple cycles from the provided DataFrame based on the given 
    cycle identifiers. Each cycle is plotted with distinct segments corresponding to different 
    statuses, and each segment is labeled accordingly.

    Parameters:
    -----------
    df_cycles : pandas.DataFrame
        The DataFrame containing cycle data. It should be multi-indexed with levels 'cycle' and 
        'status', and include 'x' and 'y' columns for plotting.

    cycle_ids : list of int
        A list of cycle identifiers to be visualized. Each identifier corresponds to a cycle 
        present in the DataFrame's 'cycle' level index.

    ax : matplotlib.axes.Axes
        The matplotlib Axes object on which the cycle data will be plotted.

    Notes:
    ------
    - The function first clears the existing content of the Axes object.
    - Only cycles present in `df_cycles` are plotted; cycles with identifiers greater than the 
      maximum available in the DataFrame are skipped.
    - Each cycle is divided into segments based on distinct 'status' values, and these segments 
      are plotted with different labels.
    - The plot is formatted with a grid, axis labels, a title, and a legend for clarity.
    """
    # Initialize
    ax.cla()
    
    # Extract points
    df_cycles = df_cycles.query(f'cycle in {cycle_ids}')
    for cycle_id in cycle_ids:
        if cycle_id > df_cycles.index.get_level_values('cycle').max():
            continue
        df_cycle = df_cycles.query(f'cycle == {cycle_id}')
        for status in df_cycle.index.get_level_values('status').unique():
            x = df_cycle.loc[(cycle_id, status), 'x']
            y = df_cycle.loc[(cycle_id, status), 'y']
            ax.plot(x, y, label=f'{cycle_id:d}, {status:s}')

    # Format plot
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Cycle {cycle_id:d}')
    ax.legend()


def visualize_peaks(df, peaks_df, sample, xcolumn, ycolumn, ax, df_zwick=None, peaks_zwick=None):
    """
    Visualize strain data with overlaid peak markers for a specific sample.

    This function plots strain data for a given sample and overlays peak markers on the plot.
    It supports displaying peaks from multiple data sources, including optional Zwick data.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing strain data indexed by sample and camera. It must include the
        specified `xcolumn` and `ycolumn`.

    peaks_df : pandas.DataFrame
        A DataFrame containing the peak data for the strains. It should include 'x' and 'y'
        columns for the peak positions, indexed by sample.

    sample : str
        The identifier for the sample to be visualized.

    xcolumn : str
        The name of the column in `df` and `df_zwick` representing the x-axis data.

    ycolumn : str
        The name of the column in `df` and `df_zwick` representing the y-axis data.

    ax : matplotlib.axes.Axes
        The matplotlib Axes object on which the strain data and peaks will be plotted.

    df_zwick : pandas.DataFrame, optional
        An optional DataFrame containing additional strain data from a Zwick source,
        indexed by sample. If provided, its data is plotted alongside the primary data.

    peaks_zwick : pandas.DataFrame, optional
        An optional DataFrame containing the peak positions for Zwick data, indexed by sample.
        If provided, its peak markers are plotted in green.

    Notes:
    ------
    - Strain data is first visualized using the `visualize_strains` function.
    - Peak markers are plotted as 'x' markers on the strain data plot, with distinct styling
      for different data sources.
    - The function resets the color cycle for the peak markers to match the strain data.
    - If `peaks_zwick` is provided, the Zwick peaks are plotted in green for distinction.
    - The plot's title is updated to indicate the visualization of cycle peaks for the given sample.
    """
    # Plot strains
    visualize_strains(df, sample, xcolumn, ycolumn, ax, df_zwick)

    # Plot peaks
    ax.set_prop_cycle(None)
    peaks_df = peaks_df.loc[sample]
    for index in peaks_df.index.unique():
        ax.plot(peaks_df.loc[index, 'x'], peaks_df.loc[index, 'y'], marker='x', ls='')

    if not peaks_zwick is None:
        peaks_zwick = peaks_zwick.loc[sample]
        ax.plot(peaks_zwick['x'], peaks_zwick['y'], marker='x', ls='', c='green')

    # Finalize plot
    ax.set_title(f'Cycle peaks of {sample}')


def export_df_to_xslx(df, file, sheet, dtype=dict()):
    """
    Export a DataFrame to an Excel file with specified settings for sheet management.

    This function writes a Pandas DataFrame to an Excel file. If the file already exists, 
    the specified sheet will be replaced; otherwise, a new file is created. This allows 
    appending or replacing sheets within an existing Excel workbook.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame to be exported to an Excel sheet.
        
    file : pathlib.Path or str
        The path to the Excel file where the DataFrame should be exported. 
        If the file exists, the function will append to it; otherwise, it will create a new file.
        
    sheet : str
        The name of the sheet within the Excel file where the DataFrame will be written.

    dtype : dict, optional
        The column (key) and data type (value) in which a column shall be written.

    Returns:
    -------
    None
        The function does not return any value. It performs the export operation and 
        prints a confirmation message upon successful export.

    Notes:
    -----
    - If the specified sheet already exists in the file, it will be replaced.
    - Ensure that the provided file path is correct and accessible for writing operations.
    - The function uses `pandas.ExcelWriter` for handling Excel file operations.

    Raises:
    ------
    ValueError
        If the DataFrame contains data types unsupported by the Excel format, this might occur 
        during the `to_excel` call.
    """
    # Writing settings
    if file.exists():
        export_kwargs = {
            'mode': 'a',
            'if_sheet_exists': 'replace'
        }
    else:
        export_kwargs = {
            'mode': 'w'
        }
    # Transform data types
    df = df.copy()
    for key in dtype.keys():
        df[key] = df[key].astype(dtype[key])
    # Write to file
    with pd.ExcelWriter(file, **export_kwargs) as writer:
        df.to_excel(writer, sheet_name=sheet)
    print(f"Successfully exported sheet '{sheet:s}' to file '{str(file):s}'.")


def insert_into_df(original_df, fill_df):
    '''
    Fill the values from one dataframe into another where they share a common index.
    If an index of original_df is not contained in fill_df, it keeps its original value.
    Input:
        original_df (pd.DataFrame) : DataFrame where values are going the be overwriten.
        fill_df     (pd.DataFrame) : DataFrame that contains the overwriting values.
    Output:
        output_df   (pd.DataFrame) : Copy of original_df where values got overwritten by
                                     fill_df if their index existed in both dataframes.
    '''
    # Reduce fill values dataframe to common index
    common_indexes = fill_df.index.intersection(original_df.index)
    fill_df = fill_df.loc[common_indexes]

    # Write fill values into default dataframe
    output_df = original_df.copy()
    output_df.loc[fill_df.index] = fill_df
    return output_df


# ---------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ FUNCTIONS FROM CREEP FITTING NOTEBOOK ------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

def find_plateaus(df, df_peaks, tol=20):
    '''
    Extract plateaus from a dataframe.
    Input:
        df       (pd.DataFrame) : DataFrame containing the time and force information.
        df_peaks (pd.DataFrame) : DataFrame containing the peaks of the time/force signal.
        tol      (float)        : Force tolerance in which force plateau is assumed.
    Output:
        df_sect  (pd.DataFrame) : DataFrame containing the force plateaus and sections between:
                                  Index 'loading'   = interval of force increase,
                                  Index 'creep'     = interval of upper plateaus,
                                  Index 'unloading' = interval of force decrease,
                                  Index 'relax'     = interval of lower plateaus.
    '''
    # Extract data
    time = df['time'].to_numpy()
    force = df['force'].to_numpy()

    # Identify cycles
    df_cycles = find_cycles(time, force, df_peaks, exclude=[]) # x equals time, y equals force
    df_cycles.sort_index(inplace=True)

    # Analyze all cycles
    tol = 20 # force tolerance where force plateau is assumed
    cycles = df_cycles.index.get_level_values('cycle').unique()
    loading_start = df_cycles.loc[0, 'index'].max()
    
    dfs_creep, dfs_relax, dfs_load, dfs_unload = [], [], [], []
    keys_creep, keys_relax, keys_load, keys_unload = [], [], [], []
    
    for cycle in cycles[1:]:
        # Extract maximum plateau
        df_max = df_cycles.loc[cycle]
        force_max = df_max['y'].max()
        df_max = df_max.query(f'y > {force_max-tol}').reset_index()

        # Extract loading section
        df_load = df_cycles.loc[cycle].query(f'index >= {loading_start} and index <= {df_max["index"].min()}').reset_index()
        
        # Append if plateau contains enough data points
        if len(df_max) > 5:
            dfs_creep.append(df_max) 
            keys_creep.append(cycle)
        dfs_load.append(df_load)
        keys_load.append(cycle)

        if cycle < cycles.max():
            # Extract minimum plateau
            df_min = pd.concat([df_cycles.loc[cycle].query('status == "unloading"'), df_cycles.loc[cycle+1].query('status == "loading"')])
            force_min = df_min['y'].quantile(0.2)
            df_min = df_min.query(f'y < {force_min+tol}').reset_index()

             # Extract unloading section
            df_unload = df_cycles.loc[cycle].query(f'index >= {df_max["index"].max()} and index <= {df_min["index"].min()}').reset_index()
            loading_start = df_min["index"].max() # loading start in next iteration
            
            # Append if plateau contains enough data points
            if len(df_min) > 5:
                dfs_relax.append(df_min)
                keys_relax.append(cycle)
            dfs_unload.append(df_unload)
            keys_unload.append(cycle)

    # Concatenate plateaus
    df_creep = pd.concat(dfs_creep, keys=keys_creep).drop_duplicates(subset='index', keep='first').drop(columns=['status'])
    df_relax = pd.concat(dfs_relax, keys=keys_relax).drop_duplicates(subset='index', keep='first').drop(columns=['status'])

    # Concatenate loading sections
    df_load = pd.concat(dfs_load, keys=keys_load).drop_duplicates(subset='index', keep='first').drop(columns=['status'])
    df_unload = pd.concat(dfs_unload, keys=keys_unload).drop_duplicates(subset='index', keep='first').drop(columns=['status'])

    # Combine results
    df_sect = pd.concat([df_load, df_creep, df_unload, df_relax], keys=['loading', 'creep', 'unloading', 'relax'], names=['status', 'cycle', 'i'])
    df_sect = df_sect.reorder_levels(['cycle', 'status', 'i']).sort_index(level='cycle', sort_remaining=False)
    return df_sect


def valid_plateaus(df_plateaus):
    '''
    Removes plateaus that do not contain all stages of a full cycle.
    Input:
        df_plateaus (pd.DataFrame) : Dataframe created with find_plateaus(df, df_peaks).
    Output
        cycle       (np.array)     : Indexes of the valid cycles.
    '''
    cycles = []
    for cycle in df_plateaus.index.get_level_values('cycle').unique():
        df = df_plateaus.loc[cycle]
        statuses = df.index.get_level_values('status').unique()
        if all([status in statuses for status in ['creep', 'unloading', 'relax']]):
            cycles.append(cycle)
    return np.array(cycles)


def get_strain_ve(times, stress, C_inv, tau, g, eps_ve):
    '''
    Calculates the viscoelastic strains at n datapoints of a KV series with m elements.
    Input:
        times  (nx1 ndarray) : Evaluated time points (must respect a sufficiently small dt and already sorted)
        stress (nx1 ndarray) : Stress matching to t
        C_inv  (nx1 ndarray) : Compliance C_inv(t) (inverse Young's modulus) matching to t
        tau    (mx1 ndarray) : KV elements' characteristic times
        g      (mx1 ndarray) : KV elements' characteristic compliances
        eps_ve (mx1 ndarray) : KV elements' starting strains; the variable is permutable
                               and will be updated with the KV elements' new strains
    Output:
        strain_ve (nx1 ndarray) : Viscoelastic strain at each time point
    '''
    # Increment over time
    strain_ve = np.zeros(len(times))
    for i in np.arange(len(times)-1):
        # Calculate strain
        strain_ve[i] = np.sum(eps_ve)

        # Calculate strain increment
        deps_ve = 1.0 / tau * ( C_inv[i] * g * stress[i] - eps_ve )
        eps_ve += deps_ve * (times[i+1]-times[i])

    # Calculate last step
    strain_ve[-1] = np.sum(eps_ve)

    return strain_ve


# ----------------------------------------------------------------------------------------------------------------------------------------
#  ------------------------------------------------- FUNCTIONS FROM PROCESSING NOTEBOOK  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def find_linear_increments(df_linear_params):
    """
    Calculate linear increments between successive cycles in a DataFrame of linear parameters.

    This function processes a DataFrame containing linear fit parameters for multiple cycles,
    identifying increments in x and y between adjacent cycles. The function computes these
    increments only for successful fits beyond the first cycle and returns a DataFrame containing
    the cumulative and incremental changes in x and y.

    Parameters:
    ----------
    df_linear_params : pandas.DataFrame
        A DataFrame containing linear fit parameters for each cycle with the following expected columns:
        - 'success_loading': A boolean indicating the success of the fit.
        - 'cycle': An integer representing the cycle number.
        - 'ymin', 'x0_loading', 'dydx_loading', 'xmin', 'xmax', 'ymax': Parameters used for linear fit calculations.

    Returns:
    -------
    pandas.DataFrame or None
        A DataFrame with the cumulative and incremental changes in x and y for each cycle, indexed by 'cycle'.
        The DataFrame contains columns:
        - 'dx': The incremental change in x.
        - 'dy': The incremental change in y.
        - 'x': The cumulative sum of x changes.
        - 'y': The cumulative sum of y changes.
        Returns None if less than two successful cycles are available for calculation.

    Raises:
    ------
    ValueError
        If an unsupported setting is encountered for the plastic increment calculation.

    Example:
    -------
    >>> df_params = pd.DataFrame({
    ...     'success_loading': [True, True, False, True],
    ...     'cycle': [0, 1, 2, 3],
    ...     'ymin': [0.1, 0.2, 0.3, 0.4],
    ...     'x0_loading': [1.0, 1.2, 1.4, 1.6],
    ...     'dydx_loading': [0.5, 0.6, 0.7, 0.8],
    ...     'xmin': [0.9, 1.1, 1.3, 1.5],
    ...     'xmax': [1.1, 1.3, 1.5, 1.7],
    ...     'ymax': [0.15, 0.25, 0.35, 0.45]
    ... })
    >>> increments_df = find_linear_increments(df_params)
    >>> print(increments_df)

    Notes:
    -----
    - The function uses a specific setting `settings['plastic_increment']` to decide the method of determining
      the increment (either 'x0' or 'xmin'). Ensure that this setting is globally defined in the environment.
    - The function assumes that all required settings and columns are correctly defined and provided.
    """
    # Initialize
    linear_fun = lambda y, dydx, x0: y / dydx + x0

    # Skip non-successful fits and first cycle
    df_linear_params = df_linear_params.query('success_loading == True and cycle != 0')
    if len(df_linear_params) < 2: # return None of less than two cycles remain
        return None
    
    # Iterate over cycles
    dfs_increment = []
    cumsum_x = 0.0
    cumsum_y = 0.0
    for cycle in range(len(df_linear_params)-1):
        # Select rows
        row_a = df_linear_params.iloc[cycle] # current row
        row_b = df_linear_params.iloc[cycle+1] # next row

        y0 = np.median([row_a['ymin'], row_b['ymin']]) # force where increment is evaluated

        # Determine increment between adjacent cycles
        match settings['plastic_increment']:
            case 'x0':
                x_a = linear_fun(y0, row_a['dydx_loading'], row_a['x0_loading'])
                x_b = linear_fun(y0, row_b['dydx_loading'], row_b['x0_loading'])
            case 'xmin':
                x_a = row_a['xmin'] if row_a['dydx_loading'] > 0 else row_a['xmax']
                x_b = row_b['xmin'] if row_b['dydx_loading'] > 0 else row_b['xmax']
            case _:
                raise ValueError(f"Unsupported increment setting '{settings['plastic_increment']}'.")
        delta_x = x_b - x_a
        delta_y = row_b['ymax'] - row_a['ymax']

        # Store results
        if cycle == 0: # store elastic range
            cumsum_y += row_a['ymax']
            dfs_increment.append( pd.DataFrame(data={'dx': 0, 'dy': row_a['ymax'], 'x': cumsum_x, 'y': cumsum_y}, index=pd.Index([cycle], name='cycle')) )

        cumsum_x += delta_x
        cumsum_y += delta_y
        dfs_increment.append( pd.DataFrame(data={'dx': delta_x, 'dy': delta_y, 'x': cumsum_x, 'y': cumsum_y}, index=pd.Index([cycle+1], name='cycle')) )

    # Return results
    return pd.concat(dfs_increment)


def df_find_linear_increments(df_elasticity):
    """
    Calculate linear increments for each unique sample within a multi-indexed DataFrame.

    This function processes a multi-indexed DataFrame, `df_elasticity`, where each top-level index
    corresponds to a unique sample identifier. It calculates linear increments for each sample
    using the `find_linear_increments` function, which processes the data for individual cycles
    within that sample. The function skips samples that do not have sufficient cycles for calculation.

    Parameters:
    ----------
    df_elasticity : pandas.DataFrame
        A multi-indexed DataFrame where the index includes a 'cycle' level. The DataFrame contains
        linear fit parameters for each cycle of different samples, which are processed separately.
        The DataFrame should have the necessary columns required by `find_linear_increments`.

    Returns:
    -------
    pandas.DataFrame
        A concatenated DataFrame containing the linear increments for each sample. The DataFrame 
        is indexed by the unique sample identifiers with 'cycle' as a sub-index. It includes columns
        such as:
        - 'dx': The incremental change in x.
        - 'dy': The incremental change in y.
        - 'x': The cumulative sum of x changes.
        - 'y': The cumulative sum of y changes.

    Notes:
    -----
    - The function relies on the `find_linear_increments` function to calculate increments for each sample.
      Ensure that `find_linear_increments` is defined and available in the environment.
    - The function assumes that the DataFrame `df_elasticity` is correctly structured with a multi-index
      including a 'cycle' level.

    Example:
    -------
    # Assuming df_elasticity is a DataFrame with a multi-index including 'cycle' and necessary columns
    >>> results_df = df_find_linear_increments(df_elasticity)
    >>> print(results_df)
    """
    # Initialize
    indexes = df_elasticity.index.droplevel('cycle').unique()
    dfs_increments, dfs_index = [], []
    
    # Determine increments over all columns, samples, and cameras separately
    for index in indexes:
        df_increments = find_linear_increments( df_elasticity.loc[index] )
        if not df_increments is None: # skip if too few cycles remained
            dfs_increments.append(df_increments)
            dfs_index.append(index)
    
    # Return results
    return pd.concat(dfs_increments, keys=dfs_index, names=indexes.names)


# ----------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS FROM ELASTICITY NOTEBOOK  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def import_xlsx(file, key='data', **kwargs):
    '''
    Assemble one dataframe from all sheets in an excel file. Requirement is
    that all imported sheets have the same structure.
    Input:
        file (Path) : Path to xlsx file
        key  (str)  : Substring the imported sheet names need to contain
        **kwargs    : Keyword arguments which are passed through to pd.read_excel(...)
    Output:
        df   (pd.DataFrame) : Dataframe with all concatenated sheets
    '''
    if file.suffix == '.ods':
        xlsx = file
        sheet_names = pd.read_excel(xlsx, sheet_name=None).keys()
        sheets = [sheet for sheet in sheet_names if key in sheet]
    else:
        xlsx = pd.ExcelFile(file, engine='openpyxl')
        sheets = [sheet for sheet in xlsx.sheet_names if key in sheet]

    dfs, index_names = [], []
    for sheet in sheets:
        df = pd.read_excel(xlsx, sheet_name=sheet, **kwargs)
        index = df.index.names if df.index.nlevels > 1 else [df.index.name]
        
        dfs.append( df )
        index_names.append( list(index) )

    df = pd.concat(dfs, keys=sheets, names=['sheet'] + index_names[0]).sort_index()
    return df


def round_to_significant_digits(number, digits=3):
    '''
    Round a number to a number of significant digits.
    Input:
        number (float) : Rounded number.
        digits (int)   : Number of significant digits.
    Output:
        Rounded number (float).
    '''
    scale = 1 if number == 0 else int(digits-(np.floor(np.log10(number)) + 1))
    number = np.round(number, scale)
    return number


def replace_multiple(s, olds, news, count=-1):
    '''
    Replace in a string multiple old values with corresponding new values.
    Input:
        s     (str)  : Input string.
        olds  (list) : List of replaced substrings.
        news  (list) : List of corresponding replacement substrings.
        count (int)  : Number of replaced occurences. -1 for all.
    Output:
        snew (str)  : Output string.
    '''
    # Iterate over all replacments
    snew = s
    for old, new in zip(olds, news):
        snew = snew.replace(old, new, count)
    return snew


def remove_outliers(a, f=1.5):
    '''
    Remove outliers based on IQR method <https://en.wikipedia.org/wiki/Interquartile_range>.
    Input:
        a     (np.ndarray) : Data array
        f     (float)      : Multiplier for the IQR method (default: 1.5)
    Output:
        a_out (np.ndarray) : Data array with outliers being removed
        index (np.ndarray) : Indexes of kept values
    '''
    # Skip if all nan
    if np.all( np.isnan(a) ):
        index = np.arange(len(a), dtype=int)
        a_out = a

    else:
        # Determine quartiles
        q1 = np.nanpercentile(a, 25)
        q3 = np.nanpercentile(a, 75)
        iqr = q3 - q1
    
        # Define outlier bounds
        bound_low = q1 - f * iqr
        bound_high = q3 + f * iqr
    
        # Remove outliers
        index = np.where((a > bound_low) & (a < bound_high))[0]
        a_out = a[index]

    return a_out, index


# ----------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS FROM PLASTICITY NOTEBOOK  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

# Define least-squares fitting
def lsq_fitting(f, xdata, ydata, p0, method='nelder-mead', options=dict()):
    '''
    Perform a least-squares fit with either scipy or lmfit.
    Input:
        f       (fun)        : Fitted function f(x, p) with parameter array p.
        xdata   (np.ndarray) : X-coordinates of fitted points.
        ydata   (np.ndarray) : Y-coordinates of fitted points.
        p0      (np.ndarray) : Initial guess of fit.
        method  (str)        : Fitting method. Either 'lmfit' or a method of scipy.optimize.minimze.
        options (dict)       : Options passed to the fitting algorithm (optional):
                               maxiter (int) : Maximum number of iterations.
                               bounds (list of tuple pairs) : Min/max of each parameter.
                               tol (float) : Tolerance for quitting minimization.
    Output:
        sol (dict) : Solution of fitting algorithm. Included keys:
                     obj     : Fitting object.
                     x       : Best fitted parameters.
                     success : Success state of fit.
                     fun     : Residual of least-squares minimization.
                     err     : Fitted parameters' standard deviation (only for method='lmfit').
    '''
    # Initialize
    sol = dict()
    
    # Get options
    maxiter = options.get('maxiter', 2000)
    bounds = options.get('bounds', None)
    tol = options.get('tol', 1e-8)
    
    # Lmfit method
    if method == 'lmfit':
        # Setup function
        pnames = [f'p{i}' for i in range(len(p0))]
        match len(p0):  # Lmfit requires manual definition of arguments, functions with *p would not work
            case 1 : fun = lambda x, p0: f(x, [p0])
            case 2 : fun = lambda x, p0, p1: f(x, [p0, p1])
            case 3 : fun = lambda x, p0, p1, p2: f(x, [p0, p1, p2])
            case 4 : fun = lambda x, p0, p1, p2, p3: f(x, [p0, p1, p2, p3])
            case 5 : fun = lambda x, p0, p1, p2, p3, p4: f(x, [p0, p1, p2, p3, p4])
            case 6 : fun = lambda x, p0, p1, p2, p3, p4, p5: f(x, [p0, p1, p2, p3, p4, p5])
            case 7 : fun = lambda x, p0, p1, p2, p3, p4, p5, p6: f(x, [p0, p1, p2, p3, p4, p5, p6])
            case 8 : fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7: f(x, [p0, p1, p2, p3, p4, p5, p6, p7])
            case 9 : fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8])
            case 10: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])
            case 11: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
            case 12: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11])
            case 13: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12])
            case 14: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13])
            case 15: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14])
            case 16: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15])
            case 17: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16])
            case 18: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17])
            case 19: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18])
            case 20: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19])
            case 21: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
            case 22: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21])
            case 23: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22])
            case 24: fun = lambda x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23: f(x, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23])
            case _: raise ValueError(f'Lmfit method is only implemented for a p0 of a length <= 24, but p0 has {len(p0)} elements.')

        # Setup model
        model = lf.Model(fun)
        params_dict = dict()
        for i, pname in enumerate(pnames):
            params_dict[pname] = {'value': p0[i]}
            if not bounds is None:
                params_dict[pname]['min'] = bounds[i][0]
                params_dict[pname]['max'] = bounds[i][1]
        params = model.make_params(**params_dict)
        result = model.fit(ydata, params, x=xdata, fit_kws=dict(ftol=tol), max_nfev=maxiter)

        # Calculate error
        pcov = result.covar  # covariance matrix
        perr = None if (pcov is None) else np.sqrt(np.diag(pcov))  # parameter standard deviation

        # Store result
        sol['obj'] = result
        sol['x'] = list(result.best_values.values())
        sol['success'] = result.success
        sol['fun'] = np.sum((result.best_fit - ydata)**2)
        sol['err'] = perr

    # Scipy method
    else:
        # Least-squares fit
        SSE = lambda p: np.sum( (f(xdata, p)-ydata)**2 )
        result = sp.optimize.minimize(SSE, x0=p0, bounds=bounds, method=method, options=dict(maxiter=maxiter), tol=tol)

        # Store result
        sol['obj'] = result
        sol['x'] = result.x
        sol['success'] = result.success
        sol['fun'] = result.fun

    # Determine R2
    SS_res = sol['fun']
    SS_tot = np.sum( ( np.nanmean(ydata) - ydata )**2 )
    sol['R2'] = 1 - SS_res / SS_tot

    # Return result
    return sol


def get_grid_coords(gs, x, y):
    '''
    Return the center points of a selected axis in a gridspec.
    Input:
        gs (mpl.gridspec.Gridspec) : Gridspec with multiple axes.
        x  (uint)                  : Horizontal index of selected axis.
        y  (uint)                  : Vertical index of selected axis.
    Output:
        xcoord (float) : Horizontal axis center in figure coordinates.
        ycoord (float) : Vertical axis center in figure coordinates.
    '''
    # Get grid positions
    fig = gs.figure
    geometry = gs.get_geometry()
    gpos = gs.get_grid_positions(fig=fig)  # (bottom, top, left, right)
    
    # Check for input error
    if x >= geometry[1] or y >= geometry[0]:
        raise ValueError(f'Invalid grid position ({(y,x)}) for grid of geometry {geometry}.')
    
    # Transform coordinates
    transform = fig.transFigure.inverted()
    
    # Select position
    xcoord = (gpos[2][x] + gpos[3][x]) / 2
    ycoord = (gpos[0][y] + gpos[1][y]) / 2

    return xcoord, ycoord

