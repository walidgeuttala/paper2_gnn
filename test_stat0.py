from scipy.stats import skew
import networkx as nx
import dgl
from data import GraphDataset
import numpy as np
from utils import *
from test_stanford_networks import *
from scipy.stats import entropy
import scipy.interpolate as interp
import random
import math
from collections import Counter
from scipy.stats import kurtosis
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def read_real_graphs(result):
    names = []
    graphs = []
    average_shortest_path = []
    with open('links.txt', 'r') as file:
        f = 0
        for line in file:
            if len(line) == 1:
                f = 1
            else :
                if f == 1:
                    f = 0
                else:
                    graph, name =  download_Stanford_network(line[:-1])
                    average_shortest_path.append(calculate_avg_shortest_path(graph))
                    graph = dgl.to_networkx(graph)
                    graph = nx.Graph(graph)
                    names.append(name[:-7])
                    graphs.append(graph)


    for file_path in result:
        graph, name = read_graph(file_path)
        graph2 = dgl.from_networkx(graph)
        average_shortest_path.append(calculate_avg_shortest_path(graph2))
        names.append(name)
        graphs.append(graph)


    for file_path in list_names:
        graph2 = read_graph2(file_path)
        graph = dgl.to_networkx(graph2)
        graph = nx.Graph(graph)
        average_shortest_path.append(calculate_avg_shortest_path(graph2))
        names.append(file_path[-1])
        graphs.append(graph)

    return graphs, names, average_shortest_path

def read_synthtic_graphs(path):
    device = 'cpu'
    dataset = GraphDataset(device=device)
    dataset.load(path)
    average_shortest_path = []
    graphs = dataset.graphs

    for idx in range(len(graphs)):
        average_shortest_path.append(calculate_avg_shortest_path(graphs[idx]))
        graphs[idx] = graphs[idx]
        graphs[idx] = dgl.to_networkx(graphs[idx])
        graphs[idx] = nx.Graph(graphs[idx])
    return graphs, average_shortest_path

def calculate_kurtosis_from_degree_list(degree_list):
    """
    Calculate the kurtosis for a given list of degrees in a graph.

    Parameters:
    - degree_list (list): List of degrees in the graph.

    Returns:
    - float: Kurtosis value.
    """
    try:
        # Convert the degree list into a degree distribution
        degree_distribution = dict(Counter(degree_list))
        
        # Calculate the kurtosis for the degree distribution
        kurtosis_value = kurtosis(list(degree_distribution.values()))
        
        return kurtosis_value
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def stats0(graphs, average_shortest_path):
    arr = [[] for _ in range(4)]
    arr[0] = average_shortest_path
    for idx, graph in enumerate(graphs):
        n = nx.number_of_nodes(graph)
        # High shortest path or no
        arr[0][idx] = math.log2(n) < arr[0][idx]
        density = nx.density(graph)
        # high or low transitivity
        arr[1].append(nx.transitivity(graph) > 10*density)
        degrees = list(dict(graph.degree()).values())
        # return True if left skewed scale free 
        arr[2].append(calculate_kurtosis_from_degree_list(degrees)>3)
        # High density or no 
        arr[3].append(density>=0.05)

    arr2 = [[] for _ in range(4)]
    arr2[0] = average_shortest_path
    for idx, graph in enumerate(graphs):
        n = nx.number_of_nodes(graph)
        # High shortest path or no
        density = nx.density(graph)
        # high or low transitivity
        arr2[1].append(nx.transitivity(graph))
        degrees = list(dict(graph.degree()).values())
        # return True if left skewed scale free 
        arr2[2].append(calculate_kurtosis_from_degree_list(degrees))
        # High density or no 
        arr2[3].append(density)
        
    return np.array(arr).T, np.array(arr2).T


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def compute_density(graphs):
    """
    Compute the density for each graph in the list.

    Parameters:
    - graphs (list): List of NetworkX graphs.

    Returns:
    - list: List of densities corresponding to each graph.
    """
    densities = [nx.density(graph) for graph in graphs]
    return np.array(densities)

def average_degree_list(graphs):
    """
    Compute the average degree for a list of NetworkX graphs.

    Parameters:
    - graphs: List of NetworkX graphs.

    Returns:
    - List of average degrees for each graph.
    """
    average_degrees = []

    for G in graphs:
        # Compute the average degree using average_degree_connectivity
        avg_degree_dict = nx.average_degree_connectivity(G)
        
        # For simplicity, just take the average of the values in the dictionary
        avg_degree = sum(avg_degree_dict.values()) / len(avg_degree_dict)
        
        average_degrees.append(avg_degree)

    return np.array(average_degrees)

def plot_histograms(arr, name, graph_type, samples):
    """
    Plot histograms for densities and save the plots.

    Parameters:
    - densities (list): List of densities.
    """
    names = ['ER_low', 'ER_high', 'WS_low', 'WS_high', 'BA_low', 'BA_high', 'Grid_low', 'Grid_high']
    # Plot histograms for each set of 250 graphs and save them
    idx = 0
    for i in range(0, len(arr), samples):
        plt.figure(figsize=(10, 5))
        plt.hist(arr[i:i+samples], bins=20, color='blue', alpha=0.7)
        plt.title(f'{name} Histogram for {names[idx]} {graph_type} Graphs')
        plt.xlabel(f'{name}')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'{name}_histogram_{names[idx]}_{graph_type}.png')  # Save the plot
        plt.close()
        idx += 1

    # Plot a general histogram for all 2000 graphs and save it
    plt.figure(figsize=(10, 5))
    plt.hist(arr, bins=40, color='green', alpha=0.7)
    plt.title(f'Overall {name} Histogram for All {graph_type} Graphs')
    plt.xlabel(f'{name}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'overall_{name}_{graph_type}_histogram.png')  # Save the plot
    plt.close()




def count_synthtic(arr, samples):
    arr = arr.reshape(-1, samples, arr.shape[1])
    return np.sum(arr, axis=1)

def average_synthtic(arr, samples):
    arr = arr.reshape(-1, samples, arr.shape[1])
    return np.mean(arr, axis=1)

result = download_and_extract(linkss)
graphs, names, average_shortest_path = read_real_graphs(result)
subnames = ['facebook_combined', 'wiki-Vote', 'p2p-Gnutella04', 'p2p-Gnutella08', 'CSphd', 'geom', 'netsience', 'adjnoun', 'football', 'hep-th', 'netscience', 'CLUSTERDataset', 'TreeGridDataset']
subname_indices = [idx for idx, name in enumerate(names) if name in subnames]
graphs = [graphs[idx] for idx in subname_indices]
average_shortest_path = [average_shortest_path[idx] for idx in subname_indices]

stat1, arr2 = stats0(graphs, average_shortest_path)


graphs, average_shortest_path = read_synthtic_graphs('data')
stat2, arr3 = stats0(graphs, average_shortest_path)
stat2 = count_synthtic(stat2, 250)
arr3 = average_synthtic(arr3, 250)

print(stat1)
print(stat2)
print(arr2)
print(arr3)

#degrees = average_degree_list(graphs)
#plot_histograms(degrees, 'Average Degree', 'Small Graphs', 250)

graphs, average_shortest_path = read_synthtic_graphs('test')
#plot_histograms(degrees, 'Average Degree', 'Medium Graphs', 50)
stat2, arr3 = stats0(graphs, average_shortest_path)
stat2 = count_synthtic(stat2, 50)
arr3 = average_synthtic(arr3, 50)

print(stat2)
print(arr3)



# Compute densities
#densities = compute_density(graphs)

#print(np.mean(densities))

# Plot histograms and save them
#plot_density_histograms(densities)
