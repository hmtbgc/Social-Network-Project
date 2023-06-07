import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp

dataset = dgl.data.CoraGraphDataset()

g = dataset[0]

u, v = g.edges()

eids = np.arange(g.num_edges())
eids = np.random.permutation(eids)


        

train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1

val_size = int(len(eids) * valid_rate)
test_size = int(len(eids) * test_rate)
train_size = len(eids) - val_size - test_size

val_and_test_eids_path = "./data/val_test_eids.txt"
with open(val_and_test_eids_path, "w") as f:
    for i in eids[:test_size+val_size]:
        print(f"{i}", file=f)

test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
val_pos_u, val_pos_v = u[eids[test_size:test_size+val_size]], v[eids[test_size:test_size+val_size]]
train_pos_u, train_pos_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.num_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size+val_size]], neg_v[neg_eids[test_size:test_size+val_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size+val_size:]], neg_v[neg_eids[test_size+val_size:]]

print(test_pos_u)
print(test_pos_v)

def write_to_file(data, path):
    u, v = data[0], data[1]
    with open(path, "w") as f:
        for i in range(u.shape[0]):
            print(f"{u[i]} {v[i]}", file=f)
            

write_to_file((test_pos_u, test_pos_v), "./data/test_pos.txt")
write_to_file((val_pos_u, val_pos_v), "./data/valid_pos.txt")
write_to_file((train_pos_u, train_pos_v), "./data/train_pos.txt")
write_to_file((test_neg_u, test_neg_v), "./data/test_neg.txt")
write_to_file((val_neg_u, val_neg_v), "./data/valid_neg.txt")
write_to_file((train_neg_u, train_neg_v), "./data/train_neg.txt")

# neighbors = []

# for i in range(g.num_nodes()):
#     u, v = g.out_edges(i)
#     v = v.tolist()
#     assert len(v) != 0
#     neighbors.append(set(v))

# # print(neighbors)
# node_number = g.num_nodes() # 2708
# edge_number = g.num_edges() # 10556
# # print(edge_number)
# edge_number_undirected = edge_number // 2


# # print(neighbors[100])
# # print(neighbors[1602])
# # print(neighbors[2056])


# for i in range(node_number):
#     for j in neighbors[i]:
#         if (i not in neighbors[j]):
#             print(f"{i} not in neighbors[{j}]")

# train_rate = 0.7
# valid_rate = 0.1
# test_rate = 0.2
# train_edge_number = int(edge_number_undirected * train_rate)
# valid_edge_number = int(edge_number_undirected * valid_rate)
# test_edge_number = edge_number_undirected - train_edge_number - valid_edge_number

# print(train_edge_number)
# print(valid_edge_number)
# print(test_edge_number)

# train_positive_edges = []
# train_negative_edges = []
# valid_positive_edges = []
# valid_negative_edges = []
# test_positive_edges = []
# test_negative_edges = []

# selected_positive_edges = set()
# selected_negative_edges = set()

# tot_number = sum([len(x) for x in neighbors])
# print(tot_number)
# i = 0
# while (i < valid_edge_number + test_edge_number): 
#     u = np.random.randint(node_number)
#     if (len(neighbors[u]) > 1):
#         temp = list(neighbors[u])
#         v = np.random.choice(temp, size=1)[0]
#         if (len(neighbors[v]) > 1):
#             u, v = max(u, v), min(u, v)
#             if ((u, v) not in selected_positive_edges):
#                 selected_positive_edges.add((u, v))
#                 if (i < valid_edge_number):
#                     valid_positive_edges.append((u, v))
#                     i += 1
#                 else:
#                     test_positive_edges.append((u, v))
#                     i += 1
#                 neighbors[u].remove(v)
#                 neighbors[v].remove(u)

# for u in range(node_number):
#     for v in neighbors[u]:
#         if (u > v):
#             if ((u, v) not in selected_positive_edges):
#                 train_positive_edges.append((u, v))
#                 selected_positive_edges.add((u, v))

# print(len(train_positive_edges))  # 3694
# print(len(valid_positive_edges))  # 527 
# print(len(test_positive_edges))   # 1057

# i = 0
# while (i < edge_number_undirected):
#     u = np.random.randint(node_number)
#     v = np.random.randint(node_number)
#     u, v = max(u, v), min(u, v)
#     if (u != v and (u, v) not in selected_positive_edges and (u, v) not in selected_negative_edges):
#         selected_negative_edges.add((u, v))
#         i += 1
#         if (i <= train_edge_number):
#             train_negative_edges.append((u, v))
#         elif (i <= train_edge_number + valid_edge_number):
#             valid_negative_edges.append((u, v))
#         else:
#             test_negative_edges.append((u, v))

# print(len(train_negative_edges)) 
# print(len(valid_negative_edges))
# print(len(test_negative_edges))

# train_positive = open("./data/train_positive.txt", "w")
# train_negative = open("./data/train_negative.txt", "w")
# valid_positive = open("./data/valid_positive.txt", "w")
# valid_negative = open("./data/valid_negative.txt", "w")
# test_positive = open("./data/test_positive.txt", "w")
# test_negative = open("./data/test_negative.txt", "w")

# for u, v in train_positive_edges:
#     print(f"{u} {v}", file=train_positive)
    
# for u, v in valid_positive_edges:
#     print(f"{u} {v}", file=valid_positive)
    
# for u, v in test_positive_edges:
#     print(f"{u} {v}", file=test_positive)
    
# for u, v in train_negative_edges:
#     print(f"{u} {v}", file=train_negative)
    
# for u, v in valid_negative_edges:
#     print(f"{u} {v}", file=valid_negative)
    
# for u, v in test_negative_edges:
#     print(f"{u} {v}", file=test_negative)

        
        
    



