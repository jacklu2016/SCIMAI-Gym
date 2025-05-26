from matplotlib import pyplot as plt
import networkx as nx

graph = nx.DiGraph()
# Define nodes with attributes
graph.add_nodes_from([0])  # Market
graph.add_nodes_from([1], I0=100, h=0.030)  # Retailer
graph.add_nodes_from([2], I0=110, h=0.020)  # Distributor
graph.add_nodes_from([3], I0=80, h=0.015)  # Distributor
graph.add_nodes_from([4], I0=400, C=90, o=0.010, v=1.000, h=0.012)  # Manufacturer
graph.add_nodes_from([5], I0=350, C=90, o=0.015, v=1.000, h=0.013)  # Manufacturer
graph.add_nodes_from([6], I0=380, C=80, o=0.012, v=1.000, h=0.011)  # Manufacturer
graph.add_nodes_from([7, 8])  # Raw materials
# Define edges with attributes
graph.add_edges_from([
    # Edge (1,0) connects Retailer 1 to Market 0
    (1, 0, {'p': 2.000, 'b': 0.100,  # p: price charged by retailer, b: backlog/lost sale cost
            # Function to sample demand using environment's RNG
            #'demand_dist_func': lambda **p: np_random.poisson(**p),
            # Parameters for the demand function (numpy poisson uses 'lam')
            'dist_param': {'lam': 20}}),
    (2, 1, {'L': 5, 'p': 1.500, 'g': 0.010}),  # L: Lead Time, p: purchase cost for receiver, g: pipeline holding cost
    (3, 1, {'L': 3, 'p': 1.600, 'g': 0.015}),
    (4, 2, {'L': 8, 'p': 1.000, 'g': 0.008}),
    (4, 3, {'L': 10, 'p': 0.800, 'g': 0.006}),
    (5, 2, {'L': 9, 'p': 0.700, 'g': 0.005}),
    (6, 2, {'L': 11, 'p': 0.750, 'g': 0.007}),
    (6, 3, {'L': 12, 'p': 0.800, 'g': 0.004}),
    (7, 4, {'L': 0, 'p': 0.150, 'g': 0.000}),  # L=0 means immediate transfer
    (7, 5, {'L': 1, 'p': 0.050, 'g': 0.005}),
    (8, 5, {'L': 2, 'p': 0.070, 'g': 0.002}),
    (8, 6, {'L': 0, 'p': 0.200, 'g': 0.000})
])


def plot_network(self):
    """ Plots the network structure using matplotlib. """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Cannot plot network.")
        return

    plt.figure(figsize=(12, 8))

    # Assign layers (simple heuristic based on type)
    node_layers = {}
    for node in self.graph.nodes():
        if node in self.rawmat:
            node_layers[node] = 0
        elif node in self.factory:
            node_layers[node] = 1
        # Separate distributors based on whether they are retailers
        elif node in self.retail:
            node_layers[node] = 3
        elif node in self.distrib:
            node_layers[node] = 2  # Non-retail distributors
        elif node in self.market:
            node_layers[node] = 4
        else:
            node_layers[node] = 2  # Default layer if unclassified (shouldn't happen with current logic)
    nx.set_node_attributes(self.graph, node_layers, "layer")

    # Use the 'layer' attribute for layout
    pos = nx.multipartite_layout(self.graph, subset_key='layer')

    # Draw nodes with different colors based on type
    node_colors = []
    node_labels = {}
    for node in self.graph.nodes():
        node_labels[node] = f"{node}"  # Basic label is node number
        if node in self.rawmat:
            color = 'gray'
            node_labels[node] += "\n(RawM)"
        elif node in self.factory:
            color = 'skyblue'
            node_labels[node] += "\n(Fact)"
        elif node in self.retail:
            color = 'lightgreen'
            node_labels[node] += "\n(Retail)"
        elif node in self.distrib:
            color = 'khaki'  # Non-retail distributors
            node_labels[node] += "\n(Dist)"
        elif node in self.market:
            color = 'salmon'
            node_labels[node] += "\n(Market)"
        else:
            color = 'pink'  # Unclassified?
        node_colors.append(color)

    nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=1500, alpha=0.8)
    nx.draw_networkx_edges(self.graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray', node_size=1500)
    nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=9)

    # Add edge labels (optional, can get cluttered - maybe just lead time?)
    edge_labels = {}
    for u, v, data in self.graph.edges(data=True):
        label = ""
        if 'L' in data: label += f"L={data['L']}"
        # if 'p' in data: label += f"\np={data['p']:.2f}" # Price adds clutter
        if label: edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    plt.title("Supply Network Structure")
    # Improve axis labels/titles maybe? multipartite doesn't use axes directly
    plt.text(0.5, 1.01, "Upstream (Raw Materials) -> Downstream (Market)", ha='center', transform=plt.gca().transAxes)
    plt.box(False)  # Remove frame
    plt.show()