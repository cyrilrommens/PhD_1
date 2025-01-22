'''''
File with all functions necessary to execute code in the PHD_1 repository.
Written by Cyril Rommens, 21-01-2025
Institute for Biocomputation and Physics of Complex Systems (BIFI)
Complex Systems & Networks Lab (COSNET)
University of Zaragoza
'''''


############################################ LIBRARIES #############################################

import random
import itertools
import infotopo
import xgi
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hypernetx as hnx
import networkx as nx
import seaborn as sns
import matplotlib.patches as patches
from itertools import combinations
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.utils import Bunch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



############################################ FUNCTIONS #############################################

# Optimized function to generate an empty simplicial complex
def generate_empty_simplicial_complex(num_nodes):
    """Generate an empty simplicial complex."""
    nodes = np.arange(1, num_nodes + 1)
    simplicial_complex = {}
    for r in range(1, num_nodes + 1):
        simplices = itertools.combinations(nodes, r)
        for simplex in simplices:
            simplicial_complex[simplex] = 0
    return simplicial_complex

# Optimized function to distribute values for R and S sums
def distribute_value(R_sum, S_sum, nb_of_variables):
    """Distribute values efficiently using NumPy."""
    nb_of_variables_R = random.randint(1, nb_of_variables - 1)
    nb_of_variables_S = nb_of_variables - nb_of_variables_R

    # Generate R and S distributions using numpy for efficiency
    points_R = np.sort(np.random.uniform(0, R_sum, nb_of_variables_R - 1))
    points_R = np.concatenate(([0], points_R, [R_sum]))
    distributed_values_R = np.diff(points_R)

    points_S = np.sort(np.random.uniform(0, S_sum, nb_of_variables_S - 1))
    points_S = np.concatenate(([0], points_S, [S_sum]))
    distributed_values_S = np.diff(points_S)

    # Combine, shuffle, and return the values
    distributed_values = np.concatenate([distributed_values_R, -distributed_values_S])
    np.random.shuffle(distributed_values)

    return distributed_values

# Optimized function to assign MI values
def assign_MI_values(simplicial_complex, R_sum, S_sum):
    """Assign MI values to simplicial complex efficiently."""
    count = sum(1 for key in simplicial_complex if len(key) > 2)
    distributed_values = distribute_value(R_sum, S_sum, count)
    distributed_values_iter = iter(distributed_values)

    for key in simplicial_complex:
        key_length = len(key)
        if key_length == 1:
            simplicial_complex[key] = random.uniform(2, 3)
        elif key_length == 2:
            simplicial_complex[key] = random.uniform(0, 3)
        elif key_length > 2:
            simplicial_complex[key] = next(distributed_values_iter)
    
    return simplicial_complex

# Optimized function to compute TSE complexity
def compute_tse_complexity(TC_dict):
    """
    Computes the TSE complexity given a dictionary of total correlations.
    
    Parameters:
        TC_dict (dict): Dictionary of total correlations where keys are tuples (subsets).
        
    Returns:
        float: The TSE complexity.
    """
    N = max(len(subset) for subset in TC_dict)  # Maximum subset size
    TSE = 0
    for gamma in range(1, N):
        TC_full = (gamma / N) * TC_dict[tuple(range(1, N + 1))]
        E_TC_gamma = np.mean([TC_dict[subset] for subset in TC_dict if len(subset) == gamma])
        TSE += TC_full - E_TC_gamma
    return TSE

# Convert a given pandas dataframe to a Bunch object
def dataframe_to_bunch(dataframe, target=None, feature_names=None, target_names=None, descr="Custom dataset"):
    if feature_names is None:
        feature_names = dataframe.columns.tolist()
    
    return Bunch(
        data=dataframe.to_numpy(),
        target=target if target is not None else np.zeros(dataframe.shape[0]),
        feature_names=feature_names,
        target_names=target_names if target_names is not None else ["target"],
        DESCR=descr
    )

# Coarse graining using sklearn
def coarse_grain_to_num_nodes(hypergraph, desired_num_nodes):
    """
    Coarse-grains a weighted hypergraph to a desired number of nodes.

    Parameters:
        hypergraph (dict): Dictionary where keys are hyperedges (tuples of nodes) and values are weights.
        desired_num_nodes (int): Target number of nodes after coarse-graining.

    Returns:
        dict: Coarse-grained hypergraph with grouped nodes and aggregated weights.
    """
    # Extract unique nodes from the hypergraph
    unique_nodes = sorted(set(node for edge in hypergraph for node in edge))

    # Assign nodes to groups using KMeans clustering
    node_features = np.array(unique_nodes).reshape(-1, 1)  # Simple 1D features based on node IDs
    kmeans = KMeans(n_clusters=desired_num_nodes, random_state=0).fit(node_features)
    node_groups = {node: group for node, group in zip(unique_nodes, kmeans.labels_)}

    # Initialize the coarse-grained hypergraph
    coarse_hypergraph = defaultdict(float)

    for hyperedge, weight in hypergraph.items():
        # Map nodes in the hyperedge to their groups
        coarse_hyperedge = tuple(sorted(set(node_groups[node] for node in hyperedge)))

        # Aggregate the weight for the coarse-grained hyperedge
        coarse_hypergraph[coarse_hyperedge] += abs(weight)

    return dict(coarse_hypergraph)

# Function for coarse graining with CompleX Group Interactions (XGI) library
def coarse_grain_with_xgi(hypergraph, desired_num_nodes):
    """
    Coarse-grains a weighted hypergraph to a desired number of nodes using the XGI library.

    Parameters:
        hypergraph (dict): Dictionary where keys are hyperedges (tuples of nodes) and values are weights.
        desired_num_nodes (int): Target number of nodes after coarse-graining.

    Returns:
        dict: Coarse-grained hypergraph with grouped nodes and aggregated weights.
    """
    # Create an XGI hypergraph from the input
    xgi_hypergraph = xgi.Hypergraph()
    for edge, weight in hypergraph.items():
        xgi_hypergraph.add_edge(edge, weight=weight)

    # Map nodes to groups (simple clustering based on node IDs for demonstration)
    unique_nodes = list(xgi_hypergraph.nodes)
    node_features = np.array(unique_nodes).reshape(-1, 1)  # 1D features based on node IDs

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=desired_num_nodes, random_state=0).fit(node_features)
    node_groups = {node: group for node, group in zip(unique_nodes, kmeans.labels_)}

    # Initialize coarse-grained hypergraph
    coarse_hypergraph = defaultdict(float)

    for edge_id in xgi_hypergraph.edges:
        # Get the members of the edge
        edge_members = xgi_hypergraph.edges.members(edge_id)

        # Map nodes in the hyperedge to their groups
        coarse_hyperedge = tuple(sorted(set(node_groups[node] for node in edge_members)))

        # Aggregate the weight for the coarse-grained hyperedge
        weight = xgi_hypergraph.edges[edge_id].get('weight', 1.0)
        coarse_hypergraph[coarse_hyperedge] += abs(weight)

    return dict(coarse_hypergraph)

# Function to perform simulation for a given R and S
def simulate_TSE(R_sum, S_sum, nb_of_variables, simulations_per_S_R_balance):
    """Simulate TSE calculation for a given R_sum and S_sum."""
    TSE_sum = 0
    for _ in range(simulations_per_S_R_balance):
        empty_simplicial_complex = generate_empty_simplicial_complex(nb_of_variables)
        Ninfomut_artificial = assign_MI_values(empty_simplicial_complex, R_sum, S_sum)
        TSE_value = compute_tse_complexity(Ninfomut_artificial)
        TSE_sum += TSE_value
    TSE_average = TSE_sum / simulations_per_S_R_balance
    return TSE_average

# Parallel processing for the DataFrame assignment
def fill_TSE_df(TSE_df, nb_of_variables, simulations_per_S_R_balance):
    """Fill the TSE DataFrame using parallel processing."""
    rows, cols = TSE_df.index, TSE_df.columns
    results = Parallel(n_jobs=-1)(delayed(simulate_TSE)(int(row), int(col), nb_of_variables, simulations_per_S_R_balance)
                                  for row in rows for col in cols)
    
    result_idx = 0
    for row in rows:
        for col in cols:
            TSE_df.loc[row, col] = results[result_idx]
            result_idx += 1
    return TSE_df

# Function to compute the average total correlation of a given subset
def compute_subset_tc_expectation(TC_dict, gamma):
    """
    Computes the expected total correlation for subsets of size gamma.
    
    Parameters:
        TC_dict (dict): Dictionary where keys are tuples (subsets) and values are total correlations.
        gamma (int): Size of the subset.
    
    Returns:
        float: Expected total correlation for subsets of size gamma.
    """
    subsets = [subset for subset in TC_dict if len(subset) == gamma]
    subset_tcs = [TC_dict[subset] for subset in subsets]
    return np.mean(subset_tcs) if subset_tcs else 0  # Return 0 if no subsets of size gamma exist

'''
OLD FUNCTIONS AS BACKUP

# Obtain topological information metrics using infotopo
def obtain_infotopo_metrics(dataset, dimension_max=0, dimension_tot=0):

    if dimension_max == 0:
        dimension_max = 3 #dataset.shape[1]
    if dimension_tot == 0:
        dimension_tot = 9 #dataset.shape[1]
    
    sample_size = dataset.shape[0]
    nb_of_values = 16
    forward_computation_mode = True
    work_on_transpose = False
    supervised_mode = False
    sampling_mode = 1
    deformed_probability_mode = False

    information_topo = infotopo.infotopo(dimension_max = dimension_max,
                                dimension_tot = dimension_tot,
                                sample_size = sample_size,
                                work_on_transpose = work_on_transpose,
                                nb_of_values = nb_of_values,
                                sampling_mode = sampling_mode,
                                deformed_probability_mode = deformed_probability_mode,
                                supervised_mode = supervised_mode,
                                forward_computation_mode = forward_computation_mode)
    
    Nentropie = information_topo.simplicial_entropies_decomposition(dataset)
    Ninfomut = information_topo.simplicial_infomut_decomposition(Nentropie)
    #Nfree_energy = information_topo.total_correlation_simplicial_lanscape(Nentropie)
    
    return  Nentropie, Ninfomut #, Nfree_energy

# Function to compute metrics for a time window sliding over a given timeseries
def interactions_values(df_input, window_size, window_step, dimension_max, dimension_tot):
    # Initialize an empty DataFrame
    columns = ['R_sum', 'S_sum', 'TSE_value']
    df_output = pd.DataFrame(columns=columns)

    # Set initial window conditions
    window_start = 0
    window_end = window_start + window_size

    while window_end < len(df_input):
        bunch_data = dataframe_to_bunch(df_input.iloc[window_start:window_end])
        Nentropie, Ninfomut = obtain_infotopo_metrics(bunch_data.data, dimension_max, dimension_tot)
        #Ninfomut = coarse_grain_with_xgi(Ninfomut, 4)
        S_sum = sum(value for key, value in Ninfomut.items() if len(key) > 2 and value < 0)
        R_sum = sum(value for key, value in Ninfomut.items() if len(key) > 2 and value > 0)
        TSE_value = compute_tse_complexity(Ninfomut)

        # Add the new values to the dataframe
        new_row = pd.DataFrame({'R_sum': [R_sum], 'S_sum': [S_sum], 'TSE_value': [TSE_value]})
        df_output = pd.concat([df_output, new_row], ignore_index=True)
        
        window_start += window_step
        window_end += window_step

    # Normalize and absolutize all the values so that the S and R values are all positive and the maximum R and S are 1.
    df_output.iloc[:, 1] = df_output.iloc[:, 1] * -1

    max_R = df_output.iloc[:, 0].max()
    max_S = df_output.iloc[:, 1].max()

    if max_S > 0:
        df_output.iloc[:, 1] = df_output.iloc[:, 1] / max_S        
    if max_R > 0:
        df_output.iloc[:, 0] = df_output.iloc[:, 0] / max_R

    return df_output
'''

# Obtain topological information metrics using infotopo
def obtain_infotopo_metrics(dataset, dimension_max=0, dimension_tot=0):

    if dimension_max == 0:
        dimension_max = 3 #dataset.shape[1]
    if dimension_tot == 0:
        dimension_tot = 9 #dataset.shape[1]
    
    sample_size = dataset.shape[0]
    nb_of_values = 16
    forward_computation_mode = True
    work_on_transpose = False
    supervised_mode = False
    sampling_mode = 1
    deformed_probability_mode = False

    information_topo = infotopo.infotopo(dimension_max = dimension_max,
                                dimension_tot = dimension_tot,
                                sample_size = sample_size,
                                work_on_transpose = work_on_transpose,
                                nb_of_values = nb_of_values,
                                sampling_mode = sampling_mode,
                                deformed_probability_mode = deformed_probability_mode,
                                supervised_mode = supervised_mode,
                                forward_computation_mode = forward_computation_mode)
    
    Nentropie = information_topo.simplicial_entropies_decomposition(dataset)
    Ninfomut = information_topo.simplicial_infomut_decomposition(Nentropie)
    
    return  Nentropie, Ninfomut

# Function to obtain the total correlation, equal to the free energies in the topological information metrics case
def obtain_total_correlations(Nentropie, dimension_max=0, dimension_tot=0):

    if dimension_max == 0:
        dimension_max = 3 #dataset.shape[1]
    if dimension_tot == 0:
        dimension_tot = 9 #dataset.shape[1]
    
    sample_size = len(Nentropie)
    nb_of_values = 16
    forward_computation_mode = True
    work_on_transpose = False
    supervised_mode = False
    sampling_mode = 1
    deformed_probability_mode = False

    information_topo = infotopo.infotopo(dimension_max = dimension_max,
                                dimension_tot = dimension_tot,
                                sample_size = sample_size,
                                work_on_transpose = work_on_transpose,
                                nb_of_values = nb_of_values,
                                sampling_mode = sampling_mode,
                                deformed_probability_mode = deformed_probability_mode,
                                supervised_mode = supervised_mode,
                                forward_computation_mode = forward_computation_mode)
    
    Nfree_energy = information_topo.total_correlation_simplicial_lanscape(Nentropie)
    return Nfree_energy

# Function to compute metrics for a time window sliding over a given timeseries
def interactions_values(df_input, window_size=0, window_step=0, dimension_max=0, dimension_tot=0):
    # Initialize an empty DataFrame
    columns = ['R_sum', 'S_sum', 'TSE_value']
    df_output = pd.DataFrame(columns=columns)

    # Set initial window conditions
    if window_size == 0:
        window_size = len(df_input)-1
    if window_step == 0:
        window_step = 1
    window_start = 0
    window_end = window_start + window_size

    while window_end < len(df_input):
        bunch_data = dataframe_to_bunch(df_input.iloc[window_start:window_end])
        Nentropie, Ninfomut = obtain_infotopo_metrics(bunch_data.data, dimension_max, dimension_tot)
        #Ninfomut = coarse_grain_with_xgi(Ninfomut, 4)
        S_sum = sum(value for key, value in Ninfomut.items() if len(key) > 2 and value < 0)
        R_sum = sum(value for key, value in Ninfomut.items() if len(key) > 2 and value > 0)
        TSE_value = compute_tse_complexity(Ninfomut)

        # Add the new values to the dataframe
        new_row = pd.DataFrame({'R_sum': [R_sum], 'S_sum': [S_sum], 'TSE_value': [TSE_value]})
        df_output = pd.concat([df_output, new_row], ignore_index=True)
        
        window_start += window_step
        window_end += window_step

    # Normalize and absolutize all the values so that the S and R values are all positive and the maximum R and S are 1.
    df_output.iloc[:, 1] = df_output.iloc[:, 1] * -1

    max_R = df_output.iloc[:, 0].max()
    max_S = df_output.iloc[:, 1].max()

    if max_S > 0:
        df_output.iloc[:, 1] = df_output.iloc[:, 1] / max_S        
    if max_R > 0:
        df_output.iloc[:, 0] = df_output.iloc[:, 0] / max_R

    return df_output

# Function to generate oscillator dynamics for a given number of oscillators
def simulate_coupled_oscillators(num_oscillators, dt=0.01, t_end=200, coupling_factor=0.1, output_file="Data\\time_series_coupled_oscillator.txt"):
    """
    Simulates a system of coupled oscillators.

    Parameters:
        num_oscillators (int): Number of oscillators in the system.
        dt (float): Time step for the simulation.
        t_end (float): End time for the simulation.
        initial_phases (list or np.ndarray): Initial phases of the oscillators. If None, defaults to zeros.
        natural_frequencies (list or np.ndarray): Natural frequencies of the oscillators. If None, defaults to ones.
        coupling_factor (float): Coupling strength between oscillators.
        output_file (str): Path to save the time series data.

    Returns:
        np.ndarray: Time series of the oscillators' phases.
    """
    # Set initial conditions
    pi_list = [np.pi for _ in range(num_oscillators)]
    random_numbers = [random.random() for _ in range(len(pi_list))]
    initial_phases = [value * rand for value, rand in zip(pi_list, random_numbers)]
    natural_frequencies = [random.random() for _ in range(num_oscillators)]

    # Time array
    time = np.arange(0, t_end, dt)
    num_steps = len(time)

    # Validate input lengths
    if len(initial_phases) != num_oscillators or len(natural_frequencies) != num_oscillators:
        raise ValueError("Length of initial_phases and natural_frequencies must match num_oscillators.")

    # Initialize arrays to store results
    theta_series = np.zeros((num_steps, num_oscillators))
    theta_series[0, :] = initial_phases

    # Simulate the coupled oscillator system
    for i in range(1, num_steps):
        for j in range(num_oscillators):
            # Compute coupling term for oscillator j
            coupling = sum(
                coupling_factor * np.sin(theta_series[i-1, k] - theta_series[i-1, j])
                for k in range(num_oscillators) if k != j
            )

            # Update phase using Euler's method
            theta_series[i, j] = theta_series[i-1, j] + (natural_frequencies[j] + coupling) * dt

    # Save the time series data to a text file
    header = " ".join([f"x{k+1}" for k in range(num_oscillators)])
    np.savetxt(output_file, theta_series, header=header, comments="")

    return theta_series

# Remove the top and bottom % of values from a list
def remove_extremes(data):
    """
    Removes the lowest 1% and highest 1% of values from the list.
    
    Parameters:
        data (list or numpy array): The input list of values.
    
    Returns:
        list: A list with the extreme values removed.
    """
    # Convert to a numpy array for easier percentile calculation
    data = np.array(data)
    
    # Calculate the 1st and 99th percentiles
    lower_bound = np.percentile(data, 1)
    upper_bound = np.percentile(data, 99)
    
    # Filter out values outside the range
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return filtered_data

# Function to generate a random timeseries with random values between 0 and 1 for a given number of variables
def generate_random_timeseries(num_variables, time_length, filename='Data\\timeseries_test.txt'):
    """
    Generate a random time series for a given number of variables and time length,
    and save it to a text file.
    
    Parameters:
    - num_variables (int): The number of variables (or features).
    - time_length (int): The length of the time series (number of time steps).
    - filename (str): The name of the text file to store the timeseries data (default is 'timeseries_test.txt').
    
    Returns:
    - np.ndarray: A 2D array where each row corresponds to a time step and each column corresponds to a variable.
    """
    # Generate random values between 0 and 1
    timeseries = np.random.rand(time_length, num_variables)
    
    # Save the timeseries to a text file
    np.savetxt(filename, timeseries, delimiter='\t')
    
    return timeseries