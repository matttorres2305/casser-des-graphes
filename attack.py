import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import random
import copy
import string

import time as t
from collections import defaultdict

from utils import *
from graph_model import parse_graph_to_kahip
from bridgeness import edge_bridgeness_centrality
from cut import make_cuts

def order_cfbca(edge_list:list, cut_list:list, graph, sample_percentage = 0.1, limit:int=None, dynamic=False):
    if not limit:
        limit = len(edge_list)
    cfa_dict = most_common(cut_list, score=True)
    for edge in edge_list:
        if edge not in cfa_dict.keys():
            cfa_dict[edge] = 0
    def f(edge):
        return cfa_dict[edge]*centrality_dict[edge]/len(cut_list)
    if dynamic:
        attack = []
        for i in range(limit):
            print(i)
            centrality_dict = nx.edge_betweenness_centrality(graph, k = int(len(graph.nodes)*sample_percentage), weight = 'length')
            edge_list.sort(reverse=True, key=f)
            attack.append(edge_list[0])
            graph.remove_edge(edge_list[0][0], edge_list[0][1])
            edge_list.remove(edge_list[0])
    else:
        centrality_dict = nx.edge_betweenness_centrality(graph, k = int(len(graph.nodes)*sample_percentage), weight = 'length')
        edge_list.sort(reverse=True, key=f)
        attack = edge_list[:limit]
    return attack

"""Returns the sorted list of edges according to the degree of the connected nodes."""
def order_degree(edge_list:list, graph, limit:int=None):
    if not limit:
        limit = len(edge_list)
    def f(edge):
        return graph.degree[edge[0]] + graph.degree[edge[1]]
    edge_list.sort(reverse=True, key=f)
    return edge_list[:limit]

"""Returns the sorted list of edges according to their betweenness centrality in a graph."""
def order_betweenness(edge_list:list, graph, sample_percentage = 0.1, limit:int=None, show_time = True, strong_mapping = False):
    start = t.time()
    if not limit:
        limit = len(edge_list)
    centrality_dict = nx.edge_betweenness_centrality(graph, k = int(len(graph.nodes)*sample_percentage), weight = 'length')
    def f(edge):
        return centrality_dict[edge]
    # print(centrality_dict.keys())
    if strong_mapping: 
        for edge in edge_list:
            if edge not in centrality_dict.keys():
                centrality_dict[edge] = centrality_dict[(edge[1], edge[0])]
    edge_list.sort(reverse=True, key=f)
    if show_time:
        print(f"Betweenness computed in {t.time() - start} s.")
    return edge_list[:limit]

"""Returns the sorted list of edges according to their bridgeness in a graph."""
def order_bridgeness(edge_list:list, graph, limit:int=None, show_time = True):
    start = t.time()
    if not limit:
        limit = len(edge_list)
    centrality_dict = edge_bridgeness_centrality(graph, weight = 'length')
    def f(edge):
        return centrality_dict[edge]
    edge_list.sort(reverse=True, key=f)
    if show_time:
        print(f"Bridgeness computed in {t.time() - start} s.")
    
    return edge_list[:limit]

"""Returns the sorted list of edges according to CFA in a list of cuts."""
def order_cfa(edge_list:list, cut_list:list, zero_edges:bool=False):
    cfa_list_temp = most_common(cut_list)
    cfa_list = []
    for i in range(len(cfa_list_temp)):
        if cfa_list_temp[i] in edge_list:
            cfa_list.append(cfa_list_temp[i])
    if zero_edges:
        random.shuffle(edge_list)
        for edge in edge_list:
            if edge not in cfa_list:
                cfa_list.append(edge)
    return cfa_list

"""Returns the sorted list of edges according to the minimization of graph efficiency, as well as the cost and the metric."""
def order_min_efficiency(edge_list:list, graph, limit = None, show_time = False):
    start = t.time()
    if not limit:
        limit = len(edge_list)
    attack = []
    weight_dict = nx.get_edge_attributes(graph, 'weight')
    cost_list = [0]
    norm = graph_efficiency(graph)
    efficiency_list = [1.]
    for i in range(limit):
        min_efficiency = np.inf
        argmin_efficiency = 0
        for j in range(len(edge_list)):
            temp_graph = copy.deepcopy(graph)
            temp_graph.remove_edge(edge_list[j][0], edge_list[j][1])
            efficiency = graph_efficiency(temp_graph)/norm
            if efficiency < min_efficiency:
                min_efficiency = efficiency
                argmin_efficiency = j
        attack.append(edge_list[argmin_efficiency])
        cost_list.append(weight_dict[edge_list[argmin_efficiency]])
        efficiency_list.append(efficiency)
        graph.remove_edge(edge_list[argmin_efficiency][0], edge_list[argmin_efficiency][1])
        edge_list.remove(edge_list[argmin_efficiency])
    if show_time:
        print(f"Attack with min efficiency ordering, and metrics computed in {t.time() - start} s.")
    return attack, cost_list, efficiency_list

"""Returns the value of the LCC metric after deleting the input edges of the input graph."""
def LCC_metric(graph, attack:list, LCC_norm = None):
    if not LCC_norm:
        LCC_norm = largest_connected_component_size(graph)
    for u,v in attack:
        graph.remove_edge(u,v)
    return largest_connected_component_size(graph)/LCC_norm

"""Returns the efficiency of a graph."""
def graph_efficiency(graph, show_time = False):
    start = t.time()
    n = len(graph.nodes)

    lengths = nx.all_pairs_dijkstra_path_length(graph, weight = "length")
    g_eff = 0
    for source, targets in lengths:
        for target, distance in targets.items():
            if distance > 0:
                g_eff += 1 / distance
    
    if show_time:
        print(f"Efficiency of {g_eff * 2/(n*(n-1))} computed in {t.time() - start} s.")

    return g_eff * 2/(n*(n-1))

"""Returns the list of LCC metric values for each edge removal of the input attack in order, and the corresponding cost list."""
def LCC_metric_underattack(graph_filename:str, attack_filename:str, plot_name = "", limit=None):
    G = nx.read_gml(path(graph_filename))
    weight_dict = nx.get_edge_attributes(G, 'weight')
    LCC_norm = largest_connected_component_size(G)
    attack_list = read_file(path(attack_filename, 'attacks'))
    if not limit:
        limit = len(attack_list)
    result_list = [1.]
    cost_list = [0]
    for edge in attack_list[:limit]:
        if edge not in weight_dict.keys():
            weight_dict[edge] = weight_dict[(edge[1], edge[0])]
        result_list.append(LCC_metric(G, [edge], LCC_norm))
        cost_list.append(weight_dict[edge]+cost_list[-1])

    if plot_name:
        plt.figure()
        plt.plot(cost_list, result_list)
        plt.xlabel('cost')
        plt.ylabel('LCC metric')
        plt.tight_layout()
        plt.savefig(path(plot_name), dpi = 300)

    return cost_list, result_list

"""Returns the list of efficiency values for each edge removal of the input attack in order, and the corresponding cost list."""
def efficiency_underattack(graph_filename:str, attack_filename:str, plot_name = "", limit = None):
    G = nx.read_gml(path(graph_filename))
    weight_dict = nx.get_edge_attributes(G, 'weight')
    attack_list = read_file(path(attack_filename, 'attacks'))
    for edge in attack_list:
        if edge not in weight_dict.keys():
            weight_dict[edge] = weight_dict[(edge[1], edge[0])]
    if not limit:
        limit = len(attack_list)
    result_list = [1.]
    cost_list = [0]
    eff_norm = graph_efficiency(G)
    for edge in attack_list[:limit]:
        print(f"{attack_list.index(edge)}/{limit}")
        G.remove_edge(edge[0], edge[1])
        result_list.append(graph_efficiency(G)/eff_norm)
        cost_list.append(weight_dict[edge]+cost_list[-1])
    if plot_name:
        plt.figure()
        plt.plot(cost_list, result_list)
        plt.xlabel('cost')
        plt.ylabel('efficiency')
        plt.tight_layout()
        plt.savefig(path(plot_name), dpi = 300)
    return cost_list, result_list

"""Plots the city graph with highlighted edges depending on attacks. Projection is hardcoded for Paris, and attacks and labels lists should contain 6 attacks.."""
def plot_attacks(graph_name:str, plot_name:str, attacks:list, labels:list):
    G = nx.MultiGraph(nx.read_gml(path(graph_name)))

    G.graph['crs'] = ox.settings.default_crs
    G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris
        
    edge_keys = list(G.edges)
    edgecolor_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.25)
    color_dict = {
    0 : 'blue',
    1 : 'orange',
    2 : 'green',
    3 : 'red',
    4 : 'purple',
    5 : 'brown'
    }
    custom_lines = []
    legend = labels
    for i in range(6):
        custom_lines.append(Line2D([0], [0], color=color_dict[i], lw=4))
        for edge in attacks[i]:
            edge = (edge[0], edge[1], 0)
            if edgecolor_dict[edge] == 'gray':
                edgecolor_dict[edge] = color_dict[i]
                large_dict[edge] = 2
    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(edgecolor_dict.values()), node_size=0.01, edge_linewidth=list(large_dict.values()))
    plt.legend(custom_lines, legend)
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

"""Returns a list of cuts belonging to the same chosen cluster from BIRCH clustering. Chooses the largest cluster by default."""
def get_cuts_cluster(clusters_filename:str, cuts_filename:str, cluster_id:int=0):
    clusters_list = read_file(path(clusters_filename, "clusters"))
    cuts_list = read_file(path(cuts_filename, "cuts"))
    return [cuts_list[int(id)] for id in clusters_list[cluster_id]]

"""Stores an iterated CA and records its robustness metrics results in 'attack_ca.json'."""
def iterated_cut_attack(graph_name:str, k:int, epsilon:float, iterations:int, cut_number:int, order:str, alt_result_name:str="", compute_eff:bool=True, cca:bool=False):
    city_name = graph_name.split(sep="_")[1]
    if cca:
        c = "c"
    else:
        c = ""
    ca_dict = read_json(path(f"attack_c{c}a.json"))
    if alt_result_name:
        result_name = alt_result_name
    else:
        result_name = f"attack_c{c}a.json"
    try:
        attack = read_file(path(f"attack_c{c}a_{order.lower()}_bestcut1000_k{k}_im{epsilon}_{city_name}", 'attacks'))
    except:
        try:
            cuts = read_file(path(f"cuts{cut_number}_k{k}_imb{epsilon}_{city_name}", "cuts"))
            best_cut = find_best_cuts(graph_name, cuts)[0]
            attack = []
            if order == "random":           
                random.shuffle(best_cut)
                for edge in best_cut:
                    attack.append(edge)
            else:
                print("order must be set to 'random' if no attack already.")
                sys.exit()
            write_file(attack, path(f"attack_c{c}a_{order.lower()}_bestcut1000_k{k}_im{epsilon}_{city_name}", 'attacks'))
        except:
            print("No attack, or cuts to do it instead found.")
            sys.exit()  
    G_true = nx.read_gml(path(graph_name))
    G = copy.deepcopy(G_true)
    former_dict = {}
    for edge in attack:
        G.remove_edge(edge[0], edge[1])
    for node in G.nodes:
        former_dict[node] = {"former":node}
    nx.set_node_attributes(G, former_dict)
    Gs = [G]
    for i in range(1, iterations):
        print(f"Iteration: {i+1}/{iterations}")
        newGGs = []
        for G in Gs:
            CCs = list(nx.connected_components(G))
            CCs.sort(key = len, reverse = True)
            LCCs = CCs[:2]
            for LCC in LCCs:
                G_LCC = build_graph_from_component(whole_graph=G, component=LCC, original_weight_dict=nx.get_edge_attributes(G, "weight"), original_length_dict=nx.get_edge_attributes(G, "length"), former_dict=nx.get_node_attributes(G, "former"))
                nx.write_gml(G_LCC, path(f"graph_{city_name}_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
                G_LCC = nx.read_gml(path(f"graph_{city_name}_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
                parse_graph_to_kahip(f"graph_{city_name}_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", f"kahip_{city_name}_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}")
                make_cuts(f"graph_{city_name}_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", f"kahip_{city_name}_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", cut_number, k, epsilon, alt_result_filename=f"cuts{cut_number}_{order}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}")
                cuts = read_file(path(f"cuts{cut_number}_{order}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", "cuts"))
                best_cut = find_best_cuts(f"graph_{city_name}_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", cuts)[0]
                truelabels_dict = nx.get_node_attributes(G_LCC, "former")
                if order == "BC":
                    temp_best_cut = copy.deepcopy(best_cut)
                    for j in range(len(best_cut)):
                        print(f"Betw: {j}/{len(best_cut)}")
                        edge = order_betweenness(temp_best_cut, G_LCC, limit=1)[0]
                        G_LCC.remove_edge(edge[0], edge[1])
                        attack.append((truelabels_dict[edge[0]], truelabels_dict[edge[1]]))
                        temp_best_cut.remove(edge)
                elif order == "random":
                    random.shuffle(best_cut)
                    for edge in best_cut:
                        attack.append((truelabels_dict[edge[0]], truelabels_dict[edge[1]]))
                        G_LCC.remove_edge(edge[0], edge[1])
                elif order == "reBC":
                    for edge in best_cut:
                        attack.append((truelabels_dict[edge[0]], truelabels_dict[edge[1]]))
                        G_LCC.remove_edge(edge[0], edge[1])
                elif order == "CF":
                    ordered_edges = order_cfa(best_cut, cuts, False)
                    for edge in ordered_edges:
                        attack.append((truelabels_dict[edge[0]], truelabels_dict[edge[1]]))
                        G_LCC.remove_edge(edge[0], edge[1])
                else:
                    print("order must be set to 'BC', 'random' or 'reBC'.")
                    sys.exit()
                newGGs.append(G_LCC)
        Gs = copy.deepcopy(newGGs)
    if order == "reBC":
        n = copy.deepcopy(len(attack))
        temp_attack = copy.deepcopy(attack)
        attack = []
        for i in range(n-1):
            print(f"Betw: {i}/{n}")
            edge = order_betweenness(temp_attack, G_true, limit=1, strong_mapping=True)[0]
            G_true.remove_edge(edge[0], edge[1])
            attack.append(edge)
            temp_attack.remove(edge)
        attack.append(temp_attack[0])
    write_file(attack, path(f"attack_ica_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{iterations}", 'attacks'))
    if f"{order}{iterations}" not in ca_dict["content"][city_name]["dynamic"].keys():
        ca_dict["content"][city_name]["dynamic"][f"{order}{iterations}"] = {}
    ca_dict["content"][city_name]["dynamic"][f"{order}{iterations}"][f"k={k}, imbalance={epsilon}"] = {}
    if compute_eff:
        _, ca_dict["content"][city_name]["dynamic"][f"{order}{iterations}"][f"k={k}, imbalance={epsilon}"]["efficiency"] = efficiency_underattack(graph_filename=graph_name, attack_filename=f"attack_it_{order}_bestcut{cut_number}_k{k}_im{epsilon}_it{iterations}", limit = 150)
    ca_dict["content"][city_name]["dynamic"][f"{order}{iterations}"][f"k={k}, imbalance={epsilon}"]["cost"], ca_dict["content"][city_name]["dynamic"][f"{order}{iterations}"][f"k={k}, imbalance={epsilon}"]["LCC metric"] = LCC_metric_underattack(graph_filename=graph_name, attack_filename=f"attack_it_{order.lower()}_bestcut{cut_number}_k{k}_im{epsilon}_it{iterations}")
    write_json(ca_dict, path(result_name))

"""Perform a CA using the cuts in a chosen cluster."""
def cluster_cut_attack(graph_name:str, l:int, imb:float, order:str, clusters_number:int=5, alt_name:str=""):
    cca_dict = read_json(path("attack_cca.json"))
    G = nx.read_gml(path(graph_name))
    for id in range(clusters_number):
        print(f"CA with cluster: {id}/{clusters_number}")
        try:
            cuts = get_cuts_cluster(clusters_filename=f"clusters_l{l}_imb{imb}", # _C à changer dans les fichiers...
                                    cuts_filename=f"cuts1000_k2_imb{imb}_mode2_clean",
                                    cluster_id=id)
        except:
            print("Clusters couldn't be read. Perhaps there is no clustering with this combination of parameters, or the names of clusters or cuts files were messed up.")
            sys.exit()
        best_cut = find_best_cuts(graph_name, cuts)[0]
        if order=="random":
            random.shuffle(best_cut)
            attack = best_cut
        elif order == "CF":
            attack = order_cfa(edge_list=best_cut, cut_list=cuts)
        elif order == "BC":
            G_ = copy.deepcopy(G)
            attack = []
            for i in range(len(best_cut)-1):
                print(f"Betw: {i}/{len(best_cut)}")
                edge = order_betweenness(edge_list=best_cut, graph=G_, sample_percentage=0.1, limit=1, strong_mapping = False)[0]
                attack.append(edge)
                G_.remove_edge(edge[0], edge[1])
                best_cut.remove(edge)
            attack.append(best_cut[0])
        else:
            print("order must be set to 'CF', 'BC' or 'random'.")
            sys.exit()
        write_file(attack, path(f"attack_cca_ca_{order}_imb{imb}_l{l}_cluster{id}", "attacks"))
        if order not in cca_dict["content"]["CA"][f"k=2, imbalance={imb}"][f"{l}"].keys():
            cca_dict["content"]["CA"][f"k=2, imbalance={imb}"][f"{l}"][order] = {}
        cca_dict["content"]["CA"][f"k=2, imbalance={imb}"][f"{l}"][order][f"{id}"] = {}
        cca_dict["content"]["CA"][f"k=2, imbalance={imb}"][f"{l}"][order][f"{id}"]["cost"], cca_dict["content"]["CA"][f"k=2, imbalance={imb}"][f"{l}"][order][f"{id}"]["LCC_metric"] = LCC_metric_underattack(graph_filename=graph_name,
                                                attack_filename=f"attack_cca_ca_{order}_imb{imb}_l{l}_cluster{id}",
                                                )
        _, cca_dict["content"]["CA"][f"k=2, imbalance={imb}"][f"{l}"][order][f"{id}"]["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
                                            attack_filename=f"attack_cca_ca_{order}_imb{imb}_l{l}_cluster{id}",
                                            limit = 150)
    if alt_name:
        result_name = alt_name
    else:
        result_name = 'attack_cca.json'
    write_json(cca_dict, path(result_name))

if __name__ == "__main__":
    pass

    # # Mock plot
    # G = nx.Graph()
    # alphabet = list(string.ascii_lowercase)[:23]
    # G.add_nodes_from(alphabet)
    # edges = [(0,1),(1,4),(2,3), (2,5), (3,4), (3,5), (4,6), (4,22), (5,7),(5,10),(6,7),(6,21),(7,8),(7,9),(8,12),(9,10),(9,15),(10,20),(11,12),(11,13),(12,13),(12,14),(15,16),(15,18),(18,19)]
    # alphabet_edges = []
    # for edge in edges:
    #     alphabet_edges.append((alphabet[edge[0]], alphabet[edge[1]]))
    # G.add_edges_from(alphabet_edges)
    # attack = [('f','h'),('d','e'),('e','g'),('f','k'),('h','j')]
    # norm = graph_efficiency(G)
    # for edge in attack:
    #     G.remove_edge(edge[0],edge[1])
    #     print(graph_efficiency(G)/norm)

    # # Some efficiency calculation
    # ca_dict = read_json(path("attack_ca.json"))
    # _,ca_dict["content"]["shanghai"]["dynamic"]["random3"]["k=2, imbalance=0.21"]["efficiency"] = efficiency_underattack(graph_filename="graph_shanghai_clean", attack_filename="attack_it_random_bestcut1000_k2_im0.21_shanghai", limit=150)
    # write_json(ca_dict, path("attack_ca_shanghai0.21.json"))
    
    # # random-CCA
    # cluster_cut_attack(graph_name="graph_paris_clean", l=25000, imb=0.1,
    #                  order="BC", alt_name="attack_cca_alt.json")

    # # BC-CUA
    # cut_union_attack(graph_name="graph_paris_clean", l=25000, imb=0.1,
    #                  order="BC")
    
    # # reBC-ICA
    # iterated_cut_attack(graph_name="graph_paris_clean", k=2, epsilon=0.03, iterations=3,
    #                     cut_number=1000, order="reBC")

    # BC-ICA
    iterated_cut_attack(graph_name="graph_paris_clean", k=2, epsilon=0.03, iterations=3,
                        cut_number=1000, order="BC", alt_result_name="attack_ca_bc3_0.1.json")

    # # IC-CFA
    # iterated_cut_attack(graph_name="graph_paris_clean", k=2, epsilon=0.03, iterations=3,
    #                     cut_number=1000, order="CF", compute_eff=True, alt_result_name="attack_ca.json")

    # # random-ICA
    # iterated_cut_attack(graph_name="graph_shanghai_clean", k=2, epsilon=0.21, iterations=3,
    #                     cut_number=1000, order="random", compute_eff=False)

    # # # Static degree attack 
    # graph_name = "graph_paris_clean"
    # city_name = "paris"
    # result_dict = read_json(path("attack_degree.json"))
    # G = nx.read_gml(path(graph_name))
    # m = len(G.edges)
    # attack = order_degree(edge_list=list(G.edges()), graph=G)
    # write_file(attack, path(f"attack_staticdegree_{city_name}"))
    # result_dict["content"][city_name]["static"]["cost"], result_dict["content"][city_name]["static"]["LCC metric"] = LCC_metric_underattack(graph_filename=graph_name,
    #                                             attack_filename=f"attack_staticdegree_{city_name}",
    #                                             )
    # _, result_dict["content"][city_name]["static"]["efficiency"] = efficiency_underattack(graph_filename=graph_name,
    #                                         attack_filename=f"attack_staticdegree_{city_name}",
    #                                         limit = 150)
    # write_json(result_dict, path("attack_degree.json"))

    # # Dynamic degree attack 
    # graph_name = "graph_paris_clean"
    # city_name = "paris"
    # result_dict = read_json(path("attack_degree.json"))["content"]
    # # G = nx.read_gml(path(graph_name))
    # # m = len(G.edges)
    # # attack = []
    # # for i in range(m):
    # #     if (i % 100) == 0:
    # #         print(i)
    # #     edge = order_degree(edge_list=list(G.edges()), graph=G, limit=1)[0]
    # #     attack.append(edge)
    # #     G.remove_edge(edge[0], edge[1])
    # # write_file(attack, path(f"attack_dynamicdegree_{city_name}"))
    # result_dict[city_name]["dynamic"]["cost"], result_dict[city_name]["dynamic"]["LCC metric"] = LCC_metric_underattack(graph_filename=graph_name,
    #                                             attack_filename=f"attack_dynamicdegree_{city_name}",
    #                                             )
    # _, result_dict[city_name]["dynamic"]["efficiency"] = efficiency_underattack(graph_filename=graph_name,
    #                                         attack_filename=f"attack_dynamicdegree_{city_name}",
    #                                         limit = 150)
    # write_json(result_dict, path("attack_degree.json"))

    # # Dynamic CFA
    # n = 200
    # k = 2
    # imb = 0.1
    # cut_number = 100
    # G = nx.read_gml(path("graph_paris_clean"))
    # nx.write_gml(G, path("temp_graph_dyncfa"))
    # parse_graph_to_kahip("temp_graph_dyncfa", "temp_kahip_dyncfa")
    # attack = []
    # for i in range(n):
    #     print(f"dyn CFA : {i}/{n}")
    #     G = nx.read_gml(path("temp_graph_dyncfa"))
    #     make_cuts(graph_name="temp_graph_dyncfa", kahip_graph="temp_kahip_dyncfa",
    #               cut_number=cut_number, k=k, imbalance=imb,
    #               alt_result_filename="temp_cuts_dyncfa")
    #     cuts = read_file(path("temp_cuts_dyncfa"))
    #     attack.append(order_cfa(edge_list=G.edges, cut_list=cuts)[0])
    #     G.remove_edge(attack[-1][0], attack[-1][1])
    #     try:
    #         nx.write_gml(G, path("temp_graph_dyncfa"))
    #         parse_graph_to_kahip("temp_graph_dyncfa", "temp_kahip_dyncfa")
    #     except:
    #         CCs = list(nx.connected_components(G))
    #         CCs.sort(key = len, reverse = True)
    #         LCC = CCs[0]
    #         G_LCC = build_graph_from_component(whole_graph=G, component=LCC, original_weight_dict=nx.get_edge_attributes(G, "weight"), original_length_dict=nx.get_edge_attributes(G, "length"), former_dict=nx.get_node_attributes(G, "former"))
    #         nx.write_gml(G_LCC, path("temp_graph_dyncfa"))
    #         parse_graph_to_kahip("temp_graph_dyncfa", "temp_kahip_dyncfa")
    # write_file(attack, path(f"attack_cfa_dyn{n}_cuts{cut_number}_k{k}_imb{imb}_paris"))
    # print("Computing robustness metrics.")
    # cfa_dict = read_json(path("attack_cfa.json"))
    # cfa_dict["content"]["paris"]["dynamic"][f"k={k}, imbalance={imb}"] = {}
    # cfa_dict["content"]["paris"]["dynamic"][f"k={k}, imbalance={imb}"]["cost"], cfa_dict["content"]["paris"]["dynamic"][f"k={k}, imbalance={imb}"]["LCC metric"] = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_dyn{n}_cuts{cut_number}_k{k}_imb{imb}_paris",
    #                                             )
    # _, cfa_dict["content"]["paris"]["dynamic"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_dyn{n}_cuts{cut_number}_k{k}_imb{imb}_paris",
    #                                             limit = min(n, 150))
    # write_json(cfa_dict, path("attack_cfa.json"))
    # os.remove(path("temp_graph_dyncfa"))
    # os.remove(path("temp_kahip_dyncfa"))
    # os.remove(path("temp_cuts_dyncfa"))

    # # Plot of CCFA and CFA on the Paris graph
    # l = 30000
    # cluster_list = read_file(path(f"clusters_birch_C_md{l}_clean0.03"))
    # attacks = [read_file(path("attack_cfa_cuts1000_k2_imb0.3_mode2_clean"))[:150]]
    # labels = ["CFA"]
    # for i in range(5):
    #     attacks.append(read_file(path(f"attack_cfa_cluster{i}_md{l}")))
    #     labels.append(f"CCFA: {len(cluster_list[i])}")
    # plot_attacks(graph_name="graph_paris_clean",
    #              plot_name=f"attack_ccfa_l{l}_paris.png",
    #              attacks=attacks,
    #              labels=labels)

    # # Dynamic CA with betweenness ordering
    # tag = "BC2"
    # graph_name = "graph_paris_clean"
    # k = 2
    # imb = 0.1
    # dyn_cut_number = 1000
    # ca_dict = read_json(path("attack_ca_new.json"))
    # attack = read_file(path(f"attack_ca_bc_bestcut1000_k{k}_im{imb}"))
    # first_lim = len(attack)
    # G = nx.read_gml(path(graph_name))
    # norm = graph_efficiency(G)
    # for edge in attack:
    #     G.remove_edge(edge[0], edge[1])
    # nx.write_gml(G,path("temp_brokengraph"))
    # CCs = list(nx.connected_components(G))
    # CCs.sort(key = len, reverse = True)
    # LCC = CCs[0]
    # newG = build_graph_from_component(whole_graph=G, component=LCC, original_weight_dict=nx.get_edge_attributes(G, "weight"), original_length_dict=nx.get_edge_attributes(G, "length"))
    # truelabels_dict = nx.get_node_attributes(newG, "former")
    # nx.write_gml(newG, path("temp_graph_dynamicca"))
    # parse_graph_to_kahip("temp_graph_dynamicca", "temp_kahip_dynamicca")
    # make_cuts("temp_graph_dynamicca", "temp_kahip_dynamicca", dyn_cut_number, k, imb, alt_result_filename="temp_cuts_dynamicca")
    # cuts = read_file(path("temp_cuts_dynamicca"))
    # best_cut = find_best_cuts("temp_graph_dynamicca", cuts)[0]
    # attack = []
    # temp_best_cut = best_cut
    # for i in range(len(best_cut)-1):
    #     print(f"CA ordered with betw: {i}/{len(best_cut)}")
    #     edge = order_betweenness(edge_list=temp_best_cut, graph=newG, sample_percentage=0.1, limit=1, strong_mapping = False)[0]
    #     attack.append(edge)
    #     newG.remove_edge(int(edge[0]), int(edge[1]))
    #     temp_best_cut.remove(edge)
    # attack.append(temp_best_cut[0])
    # true_attack = []
    # for edge in attack:
    #     true_attack.append((truelabels_dict[int(edge[0])], truelabels_dict[int(edge[1])]))
    #     print(true_attack[-1] in G.edges)
    # write_file(true_attack, path(f"temp_attack_dynamicca"))
    # cost, eff = efficiency_underattack(graph_filename="temp_brokengraph",
    #                                         attack_filename=f"temp_attack_dynamicca")
    # true_cost = ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["cost"] + [cost[i] + ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["cost"][-1] for i in range(1, len(cost))]
    # true_eff = ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"] + [eff[i] * ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"][-1] for i in range(1, len(eff))]
    # ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["efficiency"] = true_cost, true_eff
    # write_json(ca_dict, path("attack_ca_new.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for key in ca_dict["content"]["paris"]["dynamic"]["random2"].keys():
    #     key_ = key.split(", ")
    #     k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
    #     plt.plot(ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["random2"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'dyn-random2 CA: k={k}, imbalance={imb}')
    # for key in ca_dict["content"]["paris"]["dynamic"]["BC2"].keys():
    #     key_ = key.split(", ")
    #     k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
    #     plt.plot(ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'dyn-BC2 CA: k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_dyn_randomvsbc_random_bestcut1000_efficiency.png"), dpi = 300)
    # plt.close()
    # os.remove(path("temp_graph_dynamicca"))
    # os.remove(path("temp_kahip_dynamicca"))
    # os.remove(path("temp_cuts_dynamicca"))
    # os.remove(path("temp_attack_dynamicca"))
    # os.remove(path("temp_brokengraph"))

    
    # # # n-iterated CA with BC ordering
    # graph_name = "graph_paris_clean"
    # k = 2
    # imb = 0.03
    # iterations = 3
    # dyn_cut_number = 1000
    # ca_dict = {}
    # # attack = read_file(path(f"attack_ca_bc_bestcut1000_k{k}_im{imb}", 'attacks'))
    # # G_true = nx.read_gml(path(graph_name))
    # # for edge in attack:
    # #     G_true.remove_edge(edge[0], edge[1])
    # # former_dict = {}
    # # for node in G_true.nodes:
    # #     former_dict[node] = {"former":node}
    # # nx.set_node_attributes(G_true, former_dict)
    # # nx.write_gml(G_true,path(f"graph_paris_bestcut1000_k{k}_im{imb}"))
    # # G = copy.deepcopy(G_true)
    # # Gs = [G]
    # # for i in range(1, iterations):
    # #     print(f"Iteration: {i+1}/{iterations}")
    # #     newGGs = []
    # #     for G in Gs:
    # #         CCs = list(nx.connected_components(G))
    # #         CCs.sort(key = len, reverse = True)
    # #         LCCs = CCs[:2]
    # #         for LCC in LCCs:
    # #             G_LCC = build_graph_from_component(whole_graph=G, component=LCC, original_weight_dict=nx.get_edge_attributes(G, "weight"), original_length_dict=nx.get_edge_attributes(G, "length"), former_dict=nx.get_node_attributes(G, "former"))
    # #             nx.write_gml(G_LCC, path(f"graph_parisbc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
    # #             G_LCC = nx.read_gml(path(f"graph_parisbc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
    # #             print(largest_connected_component_size(G_LCC)/len(G_LCC.nodes))
    # #             parse_graph_to_kahip(f"graph_parisbc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", f"kahip_parisbc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}")
    # #             make_cuts(f"graph_parisbc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", f"kahip_parisbc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", dyn_cut_number, k, imb, alt_result_filename=f"cuts{dyn_cut_number}bc_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}")
    # #             cuts = read_file(path(f"cuts{dyn_cut_number}bc_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
    # #             best_cut = find_best_cuts(f"graph_parisbc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", cuts)[0]
    # #             temp_best_cut = copy.deepcopy(best_cut)
    # #             truelabels_dict = nx.get_node_attributes(G_LCC, "former")
    # #             for j in range(len(best_cut)):
    # #                 print(f"Betw: {j}/{len(best_cut)}")
    # #                 edge = order_betweenness(temp_best_cut, G_LCC, limit=1)[0]
    # #                 G_LCC.remove_edge(edge[0], edge[1])
    # #                 attack.append((truelabels_dict[edge[0]], truelabels_dict[edge[1]]))
    # #                 temp_best_cut.remove(edge)
    # #             newGGs.append(G_LCC)
    # #     Gs = copy.deepcopy(newGGs)
    # # write_file(attack, path(f"attack_it_bc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{iterations}", 'attacks'))
    # ca_dict["BC3"] = {}   
    # ca_dict["BC3"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["BC3"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["BC3"][f"k={k}, imbalance={imb}"]["LCC metric"] = LCC_metric_underattack(graph_filename=graph_name, attack_filename=f"attack_it_bc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{iterations}")
    # _, ca_dict["BC3"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename=graph_name, attack_filename=f"attack_it_bc_bestcut{dyn_cut_number}_k{k}_im{imb}_it{iterations}", limit = 150)
    # write_json(ca_dict, path("attack_ca_bis.json"))

    # # n-iterated CA with random ordering
    # graph_name = "graph_paris_clean"
    # k = 2
    # imb = 0.03
    # iterations = 3
    # dyn_cut_number = 1000
    # ca_dict = read_json(path("attack_ca.json"))
    # # attack = read_file(path(f"attack_ca_random_bestcut1000_k{k}_im{imb}", 'attacks'))
    # # G_true = nx.read_gml(path(graph_name))
    # # for edge in attack:
    # #     G_true.remove_edge(edge[0], edge[1])
    # # former_dict = {}
    # # for node in G_true.nodes:
    # #     former_dict[node] = {"former":node}
    # # nx.set_node_attributes(G_true, former_dict)
    # # nx.write_gml(G_true,path(f"graph_paris_bestcut1000_k{k}_im{imb}"))
    # # G = copy.deepcopy(G_true)
    # # Gs = [G]
    # # for i in range(1, iterations):
    # #     print(f"Iteration: {i+1}/{iterations}")
    # #     newGGs = []
    # #     for G in Gs:
    # #         CCs = list(nx.connected_components(G))
    # #         CCs.sort(key = len, reverse = True)
    # #         LCCs = CCs[:2]
    # #         for LCC in LCCs:
    # #             G_LCC = build_graph_from_component(whole_graph=G, component=LCC, original_weight_dict=nx.get_edge_attributes(G, "weight"), original_length_dict=nx.get_edge_attributes(G, "length"), former_dict=nx.get_node_attributes(G, "former"))
    # #             nx.write_gml(G_LCC, path(f"graph_paris_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
    # #             G_LCC = nx.read_gml(path(f"graph_paris_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
    # #             print(largest_connected_component_size(G_LCC)/len(G_LCC.nodes))
    # #             parse_graph_to_kahip(f"graph_paris_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", f"kahip_paris_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}")
    # #             make_cuts(f"graph_paris_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", f"kahip_paris_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", dyn_cut_number, k, imb, alt_result_filename=f"cuts{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}")
    # #             cuts = read_file(path(f"cuts{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}"))
    # #             best_cut = find_best_cuts(f"graph_paris_bestcut{dyn_cut_number}_k{k}_im{imb}_it{i+1}_G{Gs.index(G)}_LCC{LCCs.index(LCC)}", cuts)[0]
    # #             random.shuffle(best_cut)
    # #             truelabels_dict = nx.get_node_attributes(G_LCC, "former")
    # #             for edge in best_cut:
    # #                 G_LCC.remove_edge(edge[0], edge[1])
    # #                 attack.append((truelabels_dict[edge[0]], truelabels_dict[edge[1]]))
    # #             newGGs.append(G_LCC)
    # #     Gs = copy.deepcopy(newGGs)
    # # write_file(attack, path(f"attack_it_random_bestcut{dyn_cut_number}_k{k}_im{imb}_it{iterations}", 'attacks'))   
    # ca_dict["content"]["paris"]["dynamic"]["random3"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["content"]["paris"]["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["LCC metric"] = LCC_metric_underattack(graph_filename=graph_name, attack_filename=f"attack_it_random_bestcut{dyn_cut_number}_k{k}_im{imb}_it{iterations}")
    # _, ca_dict["content"]["paris"]["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename=graph_name, attack_filename=f"attack_it_random_bestcut{dyn_cut_number}_k{k}_im{imb}_it{iterations}", limit = 151)
    # write_json(ca_dict, path("attack_ca.json"))

    # # CA with cut-frequency ordering
    # ca_dict = read_json(path("attack_ca.json"))
    # k = 2
    # imb = 0.1
    # cuts = read_file(path(f"cuts1000_k{k}_imb{imb}_mode2_clean"))
    # best_cut = find_best_cuts("graph_paris_clean", cuts)[0]
    # attack = order_cfa(best_cut, cuts)
    # write_file(attack, path(f"attack_ca_cf_bestcut1000_k{k}_im{imb}"))
    # ca_dict["content"]["paris"]["static"]["CF"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["content"]["paris"]["static"]["CF"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["static"]["CF"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_ca_cf_bestcut1000_k{k}_im{imb}")
    # write_json(ca_dict, path("attack_ca.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # plt.plot(ca_dict["content"]["paris"]["static"]["CF"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["static"]["CF"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'stat-CF CA: k={k}, imbalance={imb}')
    # plt.plot(ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'stat-random CA: k={k}, imbalance={imb}')
    # plt.plot(ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["static"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'stat-BC CA: k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_stat_bestcut1000_k{k}_imb{imb}_efficiency.png"), dpi = 300)
    # plt.close()

    # # CA with max-efficiency ordering
    # ca_dict = read_json(path("attack_ca.json"))
    # k = 2
    # imb = 0.03
    # cuts = read_file(path(f"cuts1000_k{k}_imb{imb}_mode2_clean"))
    # G = nx.read_gml(path('graph_paris_clean'))
    # best_cut = find_best_cuts("graph_paris_clean", cuts)[0]
    # ca_dict["content"]["min-efficiency"] = {}
    # ca_dict["content"]["min-efficiency"][f"k={k}, imbalance={imb}"] = {}
    # attack, ca_dict["content"]["min-efficiency"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["min-efficiency"][f"k={k}, imbalance={imb}"]["efficiency"] = order_min_efficiency(best_cut, G, show_time = True)
    # write_file(attack, path(f"attack_ca_mineff_bestcut1000_k{k}_im{imb}"))
    # write_json(ca_dict, path("attack_ca.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for ord in ca_dict["content"].keys():
    #     plt.plot(ca_dict["content"][ord][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"][ord][f"k={k}, imbalance={imb}"]["efficiency"], label='CA '+ord+f': k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_orders_bestcut1000_k{k}_im{imb}_efficiency.png"), dpi = 300)
    # plt.close()

    # # CA with random ordering
    # ca_dict = read_json(path("attack_ca.json"))
    # k = 2
    # imb = 0.1
    # G = nx.read_gml(path("graph_paris_clean"))
    # cuts = read_file(path(f"cuts1000_k{k}_imb{imb}_mode2_clean", 'cuts'))
    # best_cut = find_best_cuts("graph_paris_clean", cuts)[0]
    # random.shuffle(best_cut)
    # attack = best_cut
    # write_file(attack, path(f"attack_ca_random_bestcut1000_k{k}_im{imb}", "attacks"))
    # ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_ca_random_bestcut1000_k{k}_im{imb}",
    #                                         limit = 150)
    # ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["paris"]["static"]["random"][f"k={k}, imbalance={imb}"]["LCC metric"] = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_ca_random_bestcut1000_k{k}_im{imb}")
    # write_json(ca_dict, path("attack_ca.json"))

    # # CA with BCA ordering
    # ca_dict = read_json(path("attack_ca.json"))
    # k = 4
    # imb = 0.1
    # cuts = read_file(path(f"cuts1000_k{k}_imb{imb}_mode2_clean"))
    # G = nx.read_gml(path('graph_paris_clean'))
    # best_cut = find_best_cuts("graph_paris_clean", cuts)[0]
    # attack = []
    # temp_best_cut = best_cut
    # for i in range(len(best_cut)-1):
    #     print(f"CA ordered with betw: {i}/{len(best_cut)}")
    #     edge = order_betweenness(edge_list=temp_best_cut, graph=G, sample_percentage=0.1, limit=1)[0]
    #     attack.append(edge)
    #     G.remove_edge(edge[0], edge[1])
    #     temp_best_cut.remove(edge)
    # attack.append(temp_best_cut[0])
    # write_file(attack, path(f"attack_ca_bc_bestcut1000_k{k}_im{imb}"))
    # ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"] = {}
    # ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_ca_bc_bestcut1000_k{k}_im{imb}")
    # write_json(ca_dict, path("attack_ca.json"))
    # bet_dict = read_json(path(("attack_betweenness.json")))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for key in ca_dict["content"]["BC"].keys():
    #     key_ = key.split(", ")
    #     k, imb = key_[0].split("=")[1], key_[1].split("=")[1]
    #     plt.plot(ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["content"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'CA: k={k}, imbalance={imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_bc_bestcut1000_k{k}_im{imb}_efficiency.png"), dpi = 300)
    # plt.close()

    # CCFA by cluster
    # G = nx.read_gml(path('graph_paris_clean'))
    # md = 25000
    # n=5
    # maxi=0
    # t_result_dict = {"description":"Results of CCFA with 0.1 imbalance clusters.",
    #                  "content":{md:{}}}
    # for id in range(n):
    #     result_dict = {}
    #     cuts_list = get_cuts_cluster(clusters_filename=f"clusters_birch_md{md}_clean0.1",
    #                                 cuts_filename="cuts1000_k2_imb0.1_mode2_clean",
    #                                 cluster_id=id)
    #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
    #     write_file(attack, path(f"attack_cfa_cluster{id}_md{md}_imb0.1"))
    #     result_dict["cost"], result_dict["LCC_metric"] = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cluster{id}_md{md}_imb0.1",
    #                                             )
    #     _, result_dict["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f"attack_cfa_cluster{id}_md{md}_imb0.1",
    #                                         limit = 150)
    #     t_result_dict["content"][md][id] = result_dict
    # write_json(t_result_dict, path('attack_ccfa_imb0.1.json'))
    

    # Static betweenness attack
    # G = nx.read_gml(path('graph_paris_clean'))
    # attack = order_betweenness(edge_list=list(G.edges()), graph=G, sample_percentage=0.1)
    # write_file(attack, path(f"attack_staticbetweenness"))
    # result_dict = read_file(path("attack_staticbetweenness_resultdict"))
    # # result_dict["cost"], result_dict["LCC metric"] = LCC_metric_underattack(graph_filename='graph_paris_clean',
    # #                                             attack_filename=f'attack_staticbetweenness',
    # #                                             )
    # _, result_dict["efficiency"] = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                         attack_filename=f'attack_staticbetweenness',
    #                                         limit = 150)
    # write_file(result_dict, path("attack_staticbetweenness_resultdict"))

    # # Dynamic betweenness attack and metrics
    # graph_name = "graph_paris_clean"
    # city_name = "paris"
    # result_dict = read_json(path("attack_betweenness.json"))
    # # G = nx.read_gml(path(graph_name))
    # # attack = []
    # # n = copy.deepcopy(len(G.edges()))
    # # for i in range(n):
    # #     print(f"{i}/{n}")
    # #     edge = order_betweenness(edge_list=list(G.edges()), graph=G, sample_percentage=0.1, limit=1)[0]
    # #     attack.append(edge)
    # #     G.remove_edge(edge[0], edge[1])
    # # write_file(attack, path(f"attack_bca_{city_name}"))
    # result_dict["content"][city_name]["dynamic"]["cost"], result_dict["content"][city_name]["dynamic"]["LCC metric"] = LCC_metric_underattack(graph_filename=graph_name,
    #                                             attack_filename=f"attack_bca_dyn_{city_name}",
    #                                             )
    # # # _, result_dict[city_name]["dynamic"]["efficiency"] = efficiency_underattack(graph_filename=graph_name,
    # # #                                         attack_filename=f"attack_dynamicbetweenness_{city_name}",
    # # #                                         limit = 150)
    # write_json(result_dict, path("attack_betweenness.json"))

   

    # # CFA attack for imbalance range
    # G = nx.read_gml(path('graph_paris_clean'))
    # imb_range = [0.03, 0.1, 0.16, 0.22, 0.3]
    # eff_limit = 150
    # template_start = "cuts1000_k2_imb"
    # template_end = "_mode2_clean"
    # for imb in imb_range:
    #     print(f"imb:{imb}")
    #     filename = template_start+str(imb)+template_end
    #     cuts_list = read_file(path(filename))
    #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
    #     write_file(attack, path(f"attack_cfa_cuts1000_k2_imb{imb}_mode2_clean"))
    # t_result_dict = {}
    # for imb in imb_range:
    #     print(f"imb:{imb}")
    #     result_dict = {}
    #     cost_list, LCC_list = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts1000_k2_imb{imb}_mode2_clean",
    #                                             )
    #     _, eff_list = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts1000_k2_imb{imb}_mode2_clean",
    #                                             limit = eff_limit)
    #     result_dict["cost"] = cost_list
    #     result_dict["LCC metric"] = LCC_list
    #     result_dict["efficiency"] = eff_list
    #     t_result_dict[imb] = result_dict
    # write_file(t_result_dict, path("cfa_dict_imb"+str(imb_range)))
    # # t_result_dict = read_file(path("cfa_dict_k"+str(k_range)))
    # plt.figure()
    # for imb in imb_range:
    #     plt.plot(t_result_dict[imb]["cost"], t_result_dict[imb]["LCC metric"], label=f'{imb}')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path('cfa_imbrange_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # for imb in imb_range:
    #     plt.plot(t_result_dict[imb]["cost"][:eff_limit+1], t_result_dict[imb]["efficiency"], label=f'{imb}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_imbrange_efficiency.png'), dpi = 300)
    # plt.close()

    # # # CFA attack for k range
    # G = nx.read_gml(path('graph_paris_clean'))
    # k_range = [2,3,4,5,6]
    # eff_limit = 150
    # template_start = "cuts1000_k"
    # template_end = "_imb0.03_mode2_clean"
    # # # for k in k_range:
    # # #     print(f"k:{k}")
    # # #     filename = template_start+str(k)+template_end
    # # #     cuts_list = read_file(path(filename))
    # # #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
        #       write_file(attack, path(f"attack_cfa_cuts1000_k{k}_imb0.03_mode2_clean"))
    # t_result_dict = read_file(path("cfa_dict_k"+str(k_range)))
    # for k in k_range:
    #     print(k)
    #     result_dict = t_result_dict[k]
    # #     # cost_list, LCC_list = LCC_metric_underattack(graph_filename='graph_paris_clean',
    # #     #                                         attack_filename=f'attack_cfa_cuts1000_k{k}_imb0.03_mode2_clean',
    # #     #                                         )
    #     _, eff_list = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f'attack_cfa_cuts1000_k{k}_imb0.03_mode2_clean',
    #                                             limit = eff_limit)
    # #     # result_dict["cost"] = cost_list
    # #     # result_dict["LCC metric"] = LCC_list
    #     result_dict["efficiency"] = eff_list
    #     t_result_dict[k] = result_dict
    # write_file(t_result_dict, path("cfa_dict_k"+str(k_range)))
    # # t_result_dict = read_file(path("cfa_dict_k"+str(k_range)))
    # # plt.figure()
    # # for k in k_range:
    # #     plt.plot(t_result_dict[k]["cost"], t_result_dict[k]["LCC metric"], label=f'{k}')
    # # plt.xlabel('cost')
    # # plt.ylabel('LCC metric')
    # # plt.tight_layout()
    # # plt.legend()
    # # plt.savefig(path('cfa_krange_LCC.png'), dpi = 300)
    # # plt.close()
    # plt.figure()
    # for k in k_range:
    #     plt.plot(t_result_dict[k]["cost"][:eff_limit+1], t_result_dict[k]["efficiency"], label=f'{k}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_krange_efficiency.png'), dpi = 300)
    # plt.close()

    # # CFA attack for pool range
    # G = nx.read_gml(path('graph_paris_clean'))
    # pool_range = [10,100,1000,10000]
    # eff_limit = 150
    # filename = "cuts10000_k2_imb0.03_mode2_clean"
    # for pool in pool_range:
    #     print(f"pool:{pool}")
    #     cuts_list = read_file(path(filename))[:pool]
    #     attack = order_cfa(edge_list=G.edges(), cut_list=cuts_list)
    #     write_file(attack, path(f"attack_cfa_cuts{pool}_10000_k2_imb0.03_mode2_clean"))
    # t_result_dict = {}
    # for pool in pool_range:
    #     print(f"pool:{pool}")
    #     result_dict = {}
    #     cost_list, LCC_list = LCC_metric_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts{pool}_10000_k2_imb0.03_mode2_clean",
    #                                             )
    #     _, eff_list = efficiency_underattack(graph_filename='graph_paris_clean',
    #                                             attack_filename=f"attack_cfa_cuts{pool}_10000_k2_imb0.03_mode2_clean",
    #                                             limit = eff_limit)
    #     result_dict["cost"] = cost_list
    #     result_dict["LCC metric"] = LCC_list
    #     result_dict["efficiency"] = eff_list
    #     t_result_dict[pool] = result_dict
    # write_file(t_result_dict, path("cfa_dict_pool"+str(pool_range)))
    # plt.figure()
    # for pool in pool_range:
    #     plt.plot(t_result_dict[pool]["cost"], t_result_dict[pool]["LCC metric"], label=f'{pool}')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path('cfa_poolrange_LCC.png'), dpi = 300)
    # plt.close()
    # plt.figure()
    # for pool in pool_range:
    #     plt.plot(t_result_dict[pool]["cost"][:eff_limit+1], t_result_dict[pool]["efficiency"], label=f'{pool}')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path('cfa_poolrange_efficiency.png'), dpi = 300)
    # plt.close()
    
    # best_cuts = read_file(path("cuts1000_k2_imb0.03_mode2_clean_bestcuts112"))
    # order_cut(cut=best_cuts[0],
    #           graph_name="graph_paris_clean",
    #           order_function=cfa_score,
    #           result_name="order_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts",
    #           full_cutlist_name="cuts1000_k2_imb0.03_mode2_clean")
    
    # cost_list, LCC_list = LCC_metric_underattack(graph_filename="graph_paris_clean",
    #                        attack_filename="order_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts",
    #                        plot_name="LCCmetric_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts.png")
    # write_file(cost_list, path("cost_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))
    # write_file(LCC_list, path("LCCmetric_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))

    # cost_list, eff_list = efficiency_underattack(graph_filename="graph_paris_clean",
    #                        attack_filename="order_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts",
    #                        plot_name="efficiency_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts.png")
    # # write_file(cost_list, path("cost_itbetweenness_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))
    # write_file(eff_list, path("efficiency_cfa_cuts1000_k2_imb0.03_mode2_clean_bestcuts"))