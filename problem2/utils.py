import torch
import torchvision
import random
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
from community import community_louvain

#可以随机选取多少个sample进行可视化
def plot_graph(graph,n_sample=None):
    G = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(G.nodes, n_sample)
        G = G.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    plt.figure(figsize=(9, 7))
    nx.draw_spring(G, node_size=30, arrows=False, node_color=y)
    plt.show() 


def central_and_cluster(G):
    # Centralities
    betweness_centrality = nx.betweenness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Clustering
    triangles = nx.triangles(G)
    clustering = nx.clustering(G)
    square_clustering = nx.square_clustering(G)

    x_feature=np.zeros([2708,8])
    for i,value in enumerate(x_feature):
        x_feature[i][0] = betweness_centrality[i]
        x_feature[i][1] = degree_centrality[i]
        x_feature[i][2] = eigenvector_centrality[i]
        x_feature[i][3] = closeness_centrality[i]
        x_feature[i][4] = triangles[i]
        x_feature[i][5] = clustering[i]
        x_feature[i][7] = square_clustering[i]

    return x_feature    


def importance_plot(G,graph,layout='spring_layout',part=1,method='degree',k=0.15,best=0.5,good=0.25,number=50):
    
    #根据节点的y给它们划分为不同子图，然后按照要求对其中一个进行importance_plot
    if part != None:
        y = graph.y.numpy()
        subnode = []
        for i in range(len(y)):
            if y[i] == part:
                subnode.append(i)
        G = G.subgraph(subnode)

    if layout=='spring_layout':
        pos = nx.spring_layout(G,k=k,seed=100)
    elif layout=='spectral_layout':
        pos = nx.spectral_layout(G)
    elif layout=='random_layout':
        pos = nx.random_layout(G,seed=100)
    elif layout=='shell_layout':
        pos = nx.shell_layout(G)
    elif layout=='circular_layout':
        pos = nx.circular_layout(G)

    # degree_centrality
    if method=='degree':
        degree_dict = nx.degree_centrality(G)
        max_center_value = max(degree_dict.values())
        color = []
        for i in degree_dict.keys():
            if degree_dict[i] > max_center_value*best:
                color.append("r")
            elif degree_dict[i] > max_center_value*good:
                color.append("c")
            else:
                color.append([0.5, 0.5, 0.5])  # grey
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":degree centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        # plt.show()
        return True

    # eigenvector_centrality
    elif method=='eigen':
        dict = nx.eigenvector_centrality_numpy(G)
        max_center_value = max(dict.values())
        color = []
        for i in dict.keys():
            if dict[i] > max_center_value*best:
                color.append("r")
            elif dict[i] > max_center_value*good:
                color.append("c")
            else:
                color.append([0.5, 0.5, 0.5])  # grey
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":eigenvector centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()
        return True

    # Katz_centrality
    elif method=='katz':
        w, v = np.linalg.eig(nx.to_numpy_array(G))
        rho = max(abs(w))
        alpha = 0.85/rho
        dict = nx.katz_centrality(G, alpha, beta=1)
        max_center_value = max(dict.values())
        color = []
        for i in dict.keys():
            if dict[i] > max_center_value*best:
                color.append("r")
            elif dict[i] > max_center_value*good:
                color.append("c")
            else:
                color.append([0.5, 0.5, 0.5])  # grey
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":Katz centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()
        return True

    # pagerank_centrality
    elif method=='pagerank':
        dict = nx.pagerank(G, alpha=0.85)
        max_center_value = max(dict.values())
        color = []
        for i in dict.keys():
            if dict[i] > max_center_value*best:
                color.append("r")
            elif dict[i] > max_center_value*good:
                color.append("c")
            else:
                color.append([0.5, 0.5, 0.5])  # grey
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":pagerank centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()
        return True

    # betweenness_centrality
    elif method=='between':
        dict = nx.betweenness_centrality(G)
        max_center_value = max(dict.values())
        color = []
        for i in dict.keys():
            if dict[i] > max_center_value*best:
                color.append("r")
            elif dict[i] > max_center_value*good:
                color.append("c")
            else:
                color.append([0.5, 0.5, 0.5])  # grey
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":betweenness centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()
        return True

    # closeness_centrality
    elif method=='close':
        dict = nx.closeness_centrality(G)
        max_center_value = max(dict.values())
        color = []
        for i in dict.keys():
            if dict[i] > max_center_value*best:
                color.append("r")
            elif dict[i] > max_center_value*good:
                color.append("c")
            else:
                color.append([0.5, 0.5, 0.5])  # grey
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":closeness centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()
        return True
    
    # HITS_centrality
    elif method=='hits':
        dict = nx.hits(G, max_iter=1000)
        #把dict这个turple转化为dict
        dict = dict[0]
        max_center_value = max(dict.values())
        color = []
        for i in dict.keys():
            if dict[i] > max_center_value*best:
                color.append("r")
            elif dict[i] > max_center_value*good:
                color.append("c")
            else:
                color.append([0.5, 0.5, 0.5])  # grey
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":HITS centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()
        return True
    
    elif method=='voterank':
        dict = nx.degree_centrality(G)
        color = []
        a=nx.voterank(G,number)
        for i in dict.keys():
            if i in a:
                color.append("r")
            else:
                color.append([0.5, 0.5, 0.5])
            
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("part"+str(part)+":voterank centrality")
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()          

#绘制出G的网络演变过程
def draw_evolution(G,graph, layout='spring_layout',part=None, k=0.15):
    if part != None:
        y = graph.y.numpy()
        subnode = []
        for i in range(len(y)):
            if y[i] == part:
                subnode.append(i)
        G = G.subgraph(subnode)

    if layout=='spring_layout':
        pos = nx.spring_layout(G,k=k,seed=100)
    elif layout=='spectral_layout':
        pos = nx.spectral_layout(G)
    elif layout=='random_layout':
        pos = nx.random_layout(G,seed=100)
    elif layout=='shell_layout':
        pos = nx.shell_layout(G)
    elif layout=='circular_layout':
        pos = nx.circular_layout(G)
    # 画出初始图
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part1:initial graph")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    # 画出第一次演化图
    G1 = G.copy()
    G1.remove_nodes_from(list(nx.isolates(G1)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part2:after removing isolates")
    nx.draw_networkx_nodes(G1, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G1, pos, alpha=0.5)
    plt.show()
    # 画出第二次演化图
    G2 = G.copy()
    G2.remove_nodes_from(list(nx.articulation_points(G2)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part3:after removing articulation points")
    nx.draw_networkx_nodes(G2, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G2, pos, alpha=0.5)
    plt.show()
    # 画出第三次演化图
    G3 = G2.copy()
    G3.remove_nodes_from(list(nx.k_core(G3, k=5)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part4:after removing 5-k-core")
    nx.draw_networkx_nodes(G3, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G3, pos, alpha=0.5)
    plt.show()
    # 画出第四次演化图
    G4 = G3.copy()
    G4.remove_nodes_from(list(nx.k_core(G4, k=4)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part5:after removing 4-k-core")
    nx.draw_networkx_nodes(G4, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G4, pos, alpha=0.5)
    plt.show()
    # 画出进一步演化的图
    G5 = G4.copy()
    G5.remove_nodes_from(list(nx.k_core(G5, k=3)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part6:after removing 3-k-core")
    nx.draw_networkx_nodes(G5, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G5, pos, alpha=0.5)
    plt.show()
    # 画出最终演化的图
    G6 = G5.copy()
    G6.remove_nodes_from(list(nx.k_core(G6, k=2)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part7:after removing 2-k-core")
    nx.draw_networkx_nodes(G6, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G6, pos, alpha=0.5)
    plt.show()
    # 画出最终演化的图
    G7 = G6.copy()
    G7.remove_nodes_from(list(nx.k_core(G7, k=1)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("part8:after removing 1-k-core")
    nx.draw_networkx_nodes(G7, pos, node_size=15, node_color=[0.5, 0.5, 0.5])
    nx.draw_networkx_edges(G7, pos, alpha=0.5)
    plt.show()

#构建随机规则图、ER随机图、BA无标度网络、WS小世界网络、星形网络、环形网络，节点数都为2708，并计算它们的节点间平均距离、平均群聚系数、图/社区的直径、diameter
def draw_graph():
    #构建规则图
    G1 = nx.random_graphs.random_regular_graph(20, 2708)
    #构建ER随机图
    G2 = nx.erdos_renyi_graph(2708, 0.01)
    #构建BA无标度网络
    G3 = nx.barabasi_albert_graph(2708, 10)
    #构建WS小世界网络
    G4 = nx.watts_strogatz_graph(2708, 10, 0.1)
    #构建星形网络
    G5 = nx.star_graph(2707)
    #构建环形网络
    G6 = nx.cycle_graph(2708)
    #计算规则图的节点间平均距离、平均群聚系数、图/社区的直径、diameter
    print("规则图的节点间平均距离为：", nx.average_shortest_path_length(G1))
    print("规则图的平均群聚系数为：", nx.average_clustering(G1))
    print("规则图的直径为：", nx.diameter(G1))
    print("规则图的diameter为：", nx.diameter(G1))
    #计算ER随机图的节点间平均距离、平均群聚系数、图/社区的直径、diameter
    print("ER随机图的节点间平均距离为：", nx.average_shortest_path_length(G2))
    print("ER随机图的平均群聚系数为：", nx.average_clustering(G2))
    print("ER随机图的直径为：", nx.diameter(G2))
    print("ER随机图的diameter为：", nx.diameter(G2))
    #计算BA无标度网络的节点间平均距离、平均群聚系数、图/社区的直径、diameter
    print("BA无标度网络的节点间平均距离为：", nx.average_shortest_path_length(G3))
    print("BA无标度网络的平均群聚系数为：", nx.average_clustering(G3))
    print("BA无标度网络的直径为：", nx.diameter(G3))
    print("BA无标度网络的diameter为：", nx.diameter(G3))
    #计算WS小世界网络的节点间平均距离、平均群聚系数、图/社区的直径、diameter
    print("WS小世界网络的节点间平均距离为：", nx.average_shortest_path_length(G4))
    print("WS小世界网络的平均群聚系数为：", nx.average_clustering(G4))
    print("WS小世界网络的直径为：", nx.diameter(G4))
    print("WS小世界网络的diameter为：", nx.diameter(G4))
    #计算星形网络的节点间平均距离、平均群聚系数、图/社区的直径、diameter
    print("星形网络的节点间平均距离为：", nx.average_shortest_path_length(G5))
    print("星形网络的平均群聚系数为：", nx.average_clustering(G5))
    print("星形网络的直径为：", nx.diameter(G5))
    print("星形网络的diameter为：", nx.diameter(G5))
    #计算环形网络的节点间平均距离、平均群聚系数、图/社区的直径、diameter
    print("环形网络的节点间平均距离为：", nx.average_shortest_path_length(G6))
    print("环形网络的平均群聚系数为：", nx.average_clustering(G6))
    print("环形网络的直径为：", nx.diameter(G6))
    print("环形网络的diameter为：", nx.diameter(G6))



    