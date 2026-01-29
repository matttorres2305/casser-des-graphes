import os
import sys
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import osmnx as ox
import random
import copy

import time as t
from collections import defaultdict

from utils import *

"""Plots the city graph with highlighted cuts depending on clusters or a specific cut list. Projection is hardcoded for Paris."""
def plot_best_cuts_by_cluster(graph_name:str, plot_name:str, cuts_name:str, clusters_name:str, n:int=5):
    assert n < len(list(mcolors.TABLEAU_COLORS)), "n must be inferior to 10 because we use the classic 10 matplotlib colors."
    G = nx.MultiGraph(nx.read_gml(path(graph_name)))
    G.graph['crs'] = ox.settings.default_crs
    G = ox.project_graph(G, to_crs='epsg:2154')

    clusters = read_file(path(clusters_name, 'clusters'))[:n]
    cuts = read_file(path(cuts_name, 'cuts'))
        
    edge_keys = list(G.edges)
    color_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.3)
    custom_lines = []
    legend = []
    for i in range(n):
        custom_lines.append(Line2D([0], [0], color=list(mcolors.TABLEAU_COLORS)[i], lw=4))
        legend.append(rf"CA: $i={i+1}$")
        cuts_cluster = [cuts[int(j)] for j in clusters[i]]
        best_cut = find_best_cuts(graph_name, cuts_cluster)[0]
        for edge in best_cut:
            edge = (edge[0], edge[1], 0)
            if color_dict[edge] == 'gray':
                color_dict[edge] = list(mcolors.TABLEAU_COLORS)[i]
                large_dict[edge] = 2
    plt.figure()
    ox.plot.plot_graph(G, edge_color=list(color_dict.values()), node_size=0.01, edge_linewidth=list(large_dict.values()), bgcolor = 'white')
    plt.legend(custom_lines, legend)
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

if __name__ == "__main__":
    pass
    # Plot parameters
    linewidth = 1
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    SMALL_SIZE = 10
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22
    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

    # # Targets of diverse (2,0.1)-CAs on the Paris network
    # k = 2
    # imb = 0.1
    # plot_best_cuts_by_cluster(graph_name="graph_paris_clean", plot_name=f"paris_cca_k{k}_imb{imb}.png",
    #                           cuts_name=f"cuts1000_k{k}_imb{imb}_mode2_clean",
    #                           clusters_name=f"clusters_l25000_imb{imb}")

    # # LCC of ICAs(random, 2, imb) for imb(0.1, 0.03) vs BCA
    # city = "paris"
    # metric = "LCC metric"
    # k = 2
    # order = "BC"
    # max_score = 550
    # style = {0.03:"dashed",0.1:"dashdot"}
    # label = {0.03:rf"ICA: $\epsilon=0.03$",0.1:rf"ICA: $\epsilon=0.1$"}
    # color = {0.03:"tab:blue",0.1:"tab:orange"}
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], color='black', linewidth=linewidth, linestyle='solid', label=f'BCA')
    # for imb in style.keys():
    #     score = ca_dict[f"{order}3"][f"k={k}, imbalance={imb}"][metric]
    #     plt.plot(ca_dict[f"{order}3"][f"k={k}, imbalance={imb}"]["cost"][:len(score)], score, color=color[imb], linewidth=linewidth, linestyle=style[imb], label=label[imb])
    # plt.xlabel('cost')
    # plt.ylabel(metric)
    # plt.xlim(-10, max_score)
    # plt.ylim(0, 1.02)
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ica_imb_f{order}_k{k}_{metric.split(sep=' ')[0]}.png"), dpi = 300)
    # plt.close()

    # Metric random vs CF vs BC (2,0.1)-ICAs vs BCA
    city = "paris"
    metric = "efficiency"
    k = 2
    imb = 0.1
    max_cost = 530
    min_metric = 0.3
    order_style = {"random":"dotted", "BC":"dashed", "CF":"dashdot"}
    order_label = {"random":r"ICA: $f=$random", "BC":r"ICA: $f=$BC", "CF":r"ICA: $f=$CF"}
    order_color = {"random":"tab:green", "BC":"blue", "CF":"red"}
    bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    ca_dict = read_json(path("attack_ca.json"))["content"][city]["dynamic"]
    plt.figure()
    plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], color='black', linewidth=linewidth, linestyle='solid', label=f'BCA')
    for order in order_style.keys():
        score = ca_dict[f"{order}3"][f"k={k}, imbalance={imb}"][metric]
        plt.plot(ca_dict[f"{order}3"][f"k={k}, imbalance={imb}"]["cost"][:len(score)], score, color=order_color[order], linewidth=linewidth, linestyle=order_style[order], label=order_label[order])
    plt.xlabel('cost')
    plt.ylabel(metric)
    plt.xlim(-10, max_cost)
    plt.ylim(min_metric, 1.02)
    plt.legend(loc='lower left')
    plt.tight_layout()
    metric = metric.split(sep=" ")[0]
    plt.savefig(path(f"attack_ica_f_imb{imb}_k{k}_{metric}.png"), dpi = 300)
    plt.close()

    # # Efficiency random vs CF vs BC (2,0.1)-CAs vs BCA
    # city = "paris"
    # metric = "efficiency"
    # k = 2
    # imb = 0.1
    # max_score = 210
    # order_style = {"random":"dotted", "BC":"dashed", "CF":"dashdot"}
    # order_label = {"random":r"CA: $f=$random", "BC":r"CA: $f=$BC", "CF":r"CA: $f=$CF"}
    # order_color = {"random":"tab:green", "BC":"blue", "CF":"red"}
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]["static"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], color='black', linewidth=linewidth, linestyle='solid', label=f'BCA')
    # for order in order_style.keys():
    #     score = []
    #     for a in ca_dict[order][f"k={k}, imbalance={imb}"][metric]:
    #         score.append(float(a))
    #         if float(a) < 0.7:
    #             break
    #     plt.plot(ca_dict[order][f"k={k}, imbalance={imb}"]["cost"][:len(score)], score, color=order_color[order], linewidth=linewidth, linestyle=order_style[order], label=order_label[order])
    # plt.xlabel('cost')
    # plt.ylabel(metric)
    # plt.xlim(-10, max_score)
    # plt.ylim(0.65, 1.02)
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_f_imb{imb}_{metric}.png"), dpi = 300)
    # plt.close()

    # # Efficiency 0.03 vs 0.1 (random,2)-CAs vs BCA
    # city = "paris"
    # metric = "efficiency"
    # k = 2
    # order = "random"
    # max_score = 200
    # style = {0.03:"dashed",0.1:"dashdot"}
    # label = {0.03:rf"$({k},0.03)$-CA",0.1:rf"$({k},0.1)$-CA"}
    # color = {0.03:"blue",0.1:"red"}
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]["static"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], color='black', linewidth=linewidth, linestyle='solid', label=f'BCA')
    # for imb in style.keys():
    #     score = ca_dict[f"{order}"][f"k={k}, imbalance={imb}"][metric]
    #     plt.plot(ca_dict[f"{order}"][f"k={k}, imbalance={imb}"]["cost"][:len(score)], score, color=color[imb], linewidth=linewidth, linestyle=style[imb], label=label[imb])
    # plt.xlabel('cost')
    # plt.ylabel(metric)
    # plt.xlim(-10, max_score)
    # plt.ylim(0, 1.02)
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_f{order}_k{k}_imb_{metric.split(sep=' ')[0]}.png"), dpi = 300)
    # plt.close()

    # # Efficiency of diverse (random,2,0.1)-CAs vs BCA
    # city = "paris"
    # metric = "efficiency"
    # k = 2
    # imb = 0.1
    # order = 'BC'
    # l = 25000
    # i_style = {0:"dotted", 1:"dashed", 2:"dashdot", 3:(0, (1, 10)), 4:(0, (5, 10))}
    # max_score = 210
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # cca_dict = read_json(path("attack_cca.json"))["content"]["CA"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], color='black', linewidth=linewidth, linestyle='solid', label=f'BCA')
    # for i in range(5):
    #     score = cca_dict[f"k={k}, imbalance={imb}"][f"{l}"][order][f"{i}"][metric]
    #     plt.plot(cca_dict[f"k={k}, imbalance={imb}"][f"{l}"][order][f"{i}"]["cost"][:len(score)], score, linestyle=i_style[i], linewidth=linewidth, label=rf"CA: $i={i+1}$")
    # plt.xlabel('cost')
    # plt.ylabel(metric)
    # plt.xlim(-10, max_score)
    # plt.ylim(0.65, 1.02)
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cca_fBC_imb{imb}_{metric}.png"), dpi = 300)
    # plt.close()

    # # Plot of diverse CFAs
    # city = "paris"
    # k = 2
    # imb = 0.1
    # l = 25000
    # order = 'CF'
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # cca_dict = read_json(path("attack_cca.json"))["content"][city]
    # colors = {0:"black", 1:"magenta", 2:"red", 3:"orange"}
    # for metric in ["LCC metric","efficiency"]:
    #     plt.figure()
    #     plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], label=f'BCA')
    #     cluster_list = read_file(path(f"clusters_l{l}_imb{imb}", "clusters"))
    #     for i in range(4):
    #         score = cca_dict[f"k=2, imbalance={imb}"][f"{l}"][order][f"{i}"][metric]
    #         plt.plot(cca_dict[f"k=2, imbalance={imb}"][f"{l}"][order][f"{i}"]["cost"][:len(score)], score, color = colors[i], label=rf"CFA: $i={i+1}$")
    #     plt.xlabel('cost')
    #     plt.ylabel(metric)
    #     if metric == "LCC metric":
    #         plt.xlim(-10, 550)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(path(f"attack_cfa_diversity_{metric}.png"), dpi = 300)
    #     plt.close()

    
    # # Plot of CFA vs IC-CFA vs IC-BCA
    # city = "paris"
    # k = 2
    # imb = 0.1
    # max_displayed_cost = 540
    # cfa_dict = read_json(path("attack_cfa.json"))["content"][city]["static"][f"k={k}, imbalance={imb}"]
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]["dynamic"]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # colors = {'CF':'orange', 'BC':'lime'}
    # for metric in ["LCC metric","efficiency"]:
    #     plt.figure()
    #     plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], label=f'BCA')
    #     # plt.plot(cfa_dict["cost"][:len(bet_dict[metric])], cfa_dict[metric], label=f'CFA', color='red')
    #     for order in ['CF','BC']:
    #         score = ca_dict[f"{order}3"][f"k={k}, imbalance={imb}"][metric]
    #         plt.plot(ca_dict[f"{order}3"][f"k={k}, imbalance={imb}"]["cost"][:len(score)], score, color = colors[order], label=f'IC-{order}A')
    #     plt.xlabel('cost')
    #     plt.ylabel(metric)
    #     plt.xlim(-10, min(max_displayed_cost, max(bet_dict["cost"]))+10)
    #     plt.legend()
    #     plt.tight_layout()
    #     name = metric.split(sep=" ")[0]
    #     plt.savefig(path(f"attack_cfa_iccfa_icbca_{name}_{city}.png"), dpi = 300)

    # # Plot of classical attacks
    # deg_dict = read_json(path("attack_degree.json"))["content"]["paris"]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["paris"]
    # # other_dict = read_json(path("attack_other.json"))["content"]
    # plt.figure()
    # plt.plot(bet_dict["static"]["cost"], bet_dict["static"]["LCC metric"], label=f'static-BCA')
    # plt.plot(deg_dict["static"]["cost"], deg_dict["static"]["LCC metric"], label=f'static-DA')
    # plt.plot(bet_dict["dynamic"]["cost"], bet_dict["dynamic"]["LCC metric"], label=f'dynamic-BCA')
    # plt.plot(deg_dict["dynamic"]["cost"], deg_dict["dynamic"]["LCC metric"], label=f'dynamic-DA')
    # # plt.plot(other_dict["dyn-CF-BCA"]["cost"][:50], other_dict["dyn-CF-BCA"]["LCC metric"][:50], label=f'dynamic-BC-CFA')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # # plt.xlim(-10, max(bet_dict["dynamic"]["cost"]))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_classicals_LCC.png"), dpi = 300)
    # plt.close()
    # plt.figure()
    # plt.plot(bet_dict["static"]["cost"][:151], bet_dict["static"]["efficiency"], label=f'static-BCA')
    # plt.plot(deg_dict["static"]["cost"][:151], deg_dict["static"]["efficiency"], label=f'static-DA')
    # plt.plot(bet_dict["dynamic"]["cost"][:151], bet_dict["dynamic"]["efficiency"], label=f'dynamic-BCA')
    # plt.plot(deg_dict["dynamic"]["cost"][:151], deg_dict["dynamic"]["efficiency"], label=f'dynamic-DA')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_classicals_efficiency.png"), dpi = 300)
    # plt.close()

    # # Plot of CAs
    # city = "paris"
    # k = 2
    # imb = 0.1
    # max_displayed_cost = 200
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # plt.plot(ca_dict["static"]["CF"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["static"]["CF"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'CF-CA')
    # plt.plot(ca_dict["static"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["static"]["random"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'random-CA')
    # plt.plot(ca_dict["static"]["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["static"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'BC-CA')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.xlim(-10, max_displayed_cost)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_bestcut1000_k{k}_imb{imb}_efficiency.png"), dpi = 300)
    # plt.close()

    # # Plot of ICA
    # city = "shanghai"
    # k = 2
    # max_displayed_cost = 540
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # color_dict = {0.1:"red", 0.21:"green"}
    # # imb = 0.1
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for imb in [0.1, 0.21]:
    #     eff = ca_dict["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["efficiency"]
    #     plt.plot(ca_dict["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["cost"][:len(eff)], eff, label='ICA', color = color_dict[imb])
    # # eff = ca_dict["dynamic"]["BC3"][f"k={k}, imbalance={imb}"]["efficiency"]
    # # plt.plot(ca_dict["dynamic"]["BC3"][f"k={k}, imbalance={imb}"]["cost"][:len(eff)], eff, label='IC-BCA', color = 'lime')
    # #     eff = ca_dict["dynamic"]["reBC3"][f"k={k}, imbalance={imb}"]["efficiency"]
    # #     plt.plot(ca_dict["dynamic"]["reBC3"][f"k={k}, imbalance={imb}"]["cost"][:len(eff)], eff, label=fr'reBC3-CA: $\epsilon={imb}$')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.xlim(-10, min(max_displayed_cost, max(bet_dict["cost"][:151]))+10)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ica_efficiency_shanghai.png"), dpi = 300)
    # # plt.figure()
    # # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # # for imb in [0.1, 0.21]:
    # #     eff = ca_dict["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["LCC metric"]
    # #     plt.plot(ca_dict["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["cost"][:len(eff)], eff, label=fr'ICA: $\epsilon={imb}$', color=color_dict[imb])
    # #     # if imb == 0.03:
    # #     #     ca_dict["dynamic"]["reBC3"][f"k={k}, imbalance={imb}"]["LCC metric"]
    # #     #     plt.plot(ca_dict["dynamic"]["reBC3"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["dynamic"]["reBC3"][f"k={k}, imbalance={imb}"]["LCC metric"], label=fr'reBC3-CA: $\epsilon={imb}$')
    # # plt.xlabel('cost')
    # # plt.ylabel('LCC metric')
    # # plt.xlim(-10, min(max_displayed_cost, max(bet_dict["cost"]))+10)
    # # plt.legend()
    # # plt.tight_layout()
    # # plt.savefig(path(f"attack_ica_LCC.png"), dpi = 300)
    # # plt.close()

    # # Plot of CFAs
    # city = "paris"
    # cfa_dict = read_json(path("attack_cfa.json"))["content"][city]["static"]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # color_dict = {0.03:"red", 0.1:"green"}
    # xlim = 3000
    # # # Along imbalance
    # # plt.figure()
    # # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # # for imb in [0.03, 0.1, 0.16, 0.22, 0.3]:
    # #     plt.plot(cfa_dict[f"k=2, imbalance={imb}"]["cost"][:151], cfa_dict[f"k=2, imbalance={imb}"]["efficiency"], label=fr'CFA: $\epsilon={imb}$')
    # # plt.xlabel('cost')
    # # plt.ylabel('efficiency')
    # # plt.legend()
    # # plt.tight_layout()
    # # plt.savefig(path(f"attack_cfa(imb)_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for imb in [0.03, 0.1]:
    #     plt.plot(cfa_dict[f"k=2, imbalance={imb}"]["cost"], cfa_dict[f"k=2, imbalance={imb}"]["LCC metric"], label=fr'CFA: $\epsilon={imb}$', color=color_dict[imb])
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.xlim(-10, xlim+10)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(imb)_LCC.png"), dpi = 300)
    # # # Along k
    # # plt.figure()
    # # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # # for k in [2, 3, 4, 5, 6]:
    # #     plt.plot(cfa_dict[f"k={k}, imbalance=0.03"]["cost"][:151], cfa_dict[f"k={k}, imbalance=0.03"]["efficiency"], label=fr'CFA: $k={k}$')
    # # plt.xlabel('cost')
    # # plt.ylabel('efficiency')
    # # plt.legend()
    # # plt.tight_layout()
    # # plt.savefig(path(f"attack_cfa(k)_efficiency.png"), dpi = 300)
    # # plt.figure()
    # # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # # for k in [2, 3, 4, 5, 6]:
    # #     plt.plot(cfa_dict[f"k={k}, imbalance=0.03"]["cost"], cfa_dict[f"k={k}, imbalance=0.03"]["LCC metric"], label=fr'CFA: $k={k}$')
    # # plt.xlabel('cost')
    # # plt.ylabel('LCC metric')
    # # plt.legend()
    # # plt.tight_layout()
    # # plt.savefig(path(f"attack_cfa(k)_LCC.png"), dpi = 300)
    # # plt.close()
    # # # Along n
    # # plt.figure()
    # # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # # for n in [10, 100, 1000, 10000]:
    # #     plt.plot(cfa_dict[f"n={n}"]["cost"][:151], cfa_dict[f"n={n}"]["efficiency"], label=fr'CFA: $n={n}$')
    # # plt.xlabel('cost')
    # # plt.ylabel('efficiency')
    # # plt.legend()
    # # plt.tight_layout()
    # # plt.savefig(path(f"attack_cfa(n)_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # color_dict = {10:"black", 100:"magenta", 1000:"red", 10000:"orange"}
    # for n in [10, 100, 1000, 10000]:
    #     plt.plot(cfa_dict[f"n={n}"]["cost"], cfa_dict[f"n={n}"]["LCC metric"], label=fr'CFA: $n={n}$', color=color_dict[n])
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.xlim(-10, xlim+10)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(n)_LCC.png"), dpi = 300)
    # plt.close()

    # # Plot of dynamic vs static CFA
    # city = "paris"
    # k = 2
    # imb = 0.03
    # max_displayed_costs = [350, 550] # eff, LCC
    # cfa_dict = read_json(path("attack_cfa.json"))["content"][city]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["dynamic"]
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]["static"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # plt.plot(ca_dict["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'BC-CA')
    # plt.plot(ca_dict["CF"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["CF"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'CF-CA')
    # for v in ["static", "dynamic"]:
    #     plt.plot(cfa_dict[v][f"k={k}, imbalance={imb}"]["cost"][:151], cfa_dict[v][f"k={k}, imbalance={imb}"]["efficiency"], label=f'{v}-CFA')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.xlim(-10, max_displayed_costs[0])
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa_dyn_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for v in ["static", "dynamic"]:
    #     plt.plot(cfa_dict[v][f"k={k}, imbalance={imb}"]["cost"], cfa_dict[v][f"k={k}, imbalance={imb}"]["LCC metric"], label=f'{v}-CFA')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.legend()
    # plt.xlim(-10, max_displayed_costs[1])
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa_dyn_LCC.png"), dpi = 300)
    # plt.close()

    # # Plot of CCAs vs imbalance
    # city = "paris"
    # k = 2
    # l = 25000
    # order = 'BC'
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # cca_dict = read_json(path("attack_cca.json"))["content"][city]
    # for metric in ["efficiency","LCC metric"]:
    #     plt.figure()
    #     plt.plot(bet_dict["cost"][:len(bet_dict[metric])], bet_dict[metric], label=f'BCA')
    #     colors = {"0.03":{0:"red",1:"orange"}, "0.1":{0:"green",1:"lime"}}
    #     for imb in ["0.03","0.1"]:
    #         cluster_list = read_file(path(f"clusters_l{l}_imb{imb}", "clusters"))
    #         for i in range(2):
    #             score = cca_dict[f"k=2, imbalance={imb}"][f"{l}"][order][f"{i}"][metric]
    #             plt.plot(cca_dict[f"k=2, imbalance={imb}"][f"{l}"][order][f"{i}"]["cost"][:len(score)], score, color = colors[imb][i], label=rf"CC-BCA: $\epsilon={imb}$, $i={i+1}$")
    #     plt.xlabel('cost')
    #     plt.ylabel(metric)
    #     if metric == "LCC metric":
    #         plt.xlim(-10, 550)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(path(f"attack_cca_bc_{metric}.png"), dpi = 300)
    #     plt.close()

