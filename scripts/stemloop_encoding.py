from utility_functions import print_loading_bar, run_rnafold, compute_mfe
import torch
from torch_geometric.data import Data
import pandas as pd
import RNA
import numpy as np

cancer_path = 'stemloop data of cancer lncrna excel path'
non_cancer_path = 'stemloop data of non cancer lncrna excel path'

cdf = pd.read_excel(cancer_path)
ncdf = pd.read_excel(non_cancer_path)

def parse_entry(entry):

  entry = entry.to_dict()

  acession_id = entry["Accession_ID"]
  symbol = entry["Symbol"]
  gene_id = entry["Gene_ID"]
  expression = entry["Expression"]
  sequence = entry["Sequence"].upper().replace("T", "U")
  sequence_length = int(entry["Sequence_length"])
  structure = entry["Structure"]
  tot_stemloops = int(entry["Tot_stemloops"])
  stemloop_id = entry["Stemloop_id"]
  stemloop_seq = entry["Sequence_of_stemloop"].upper().replace("T", "U")
  stemloop_struct = entry["Structure_of_stemloop"]
  stemloop_len = int(entry["Seq_len_of_stemloop"])
  stem_len = int(entry["Stem_length"])
  loop_len = int(entry["Loop_length"])
  start_pos, end_pos = tuple(map(int, entry["Start_end_pos"].strip("()").split(",")))
  gc_content = round(int(entry["GC_content"])/100, 2)
  stemloop_mfe = round(float(entry["MFE_of_stem_loop"]), 3)

  return symbol, sequence_length, tot_stemloops, stemloop_id, stemloop_seq, stemloop_struct, stemloop_len, stem_len, loop_len, start_pos, gc_content, stemloop_mfe

def pos_vec_to_seq(pos_vec, seq):
  sequence = []
  for pos in pos_vec:
    sequence.append(seq[pos])
  return "".join(sequence)

def stemloop_check(struct):
  return struct.count("(") == struct.count(")")

def pos_left_stem(struct):
  pos = []
  for i in range(len(struct)):
    if struct[i] == "(":
      pos.append(i)
  return pos

def pos_right_stem(struct):
  pos = []
  for i in range(len(struct)):
    if struct[i] == ")":
      pos.append(i)
  return pos[::-1]

def compute_avg_delta_mfe(pos, seq, struct):

  nucleotides = ["A", "C", "G", "U"]
  nucleotides.remove(seq[pos])

  wt_mfe = compute_mfe(seq, struct)
  mutated_mfes = [run_rnafold(seq[:pos] + nucleotide + seq[pos+1:]) for nucleotide in nucleotides]

  avg_delta_mfe = np.mean(np.array(mutated_mfes) - wt_mfe)
  return avg_delta_mfe

def type_of_bond(source_pos, target_pos, seq, struct):

    if (struct[source_pos] == '(' and struct[target_pos] == ')') or (struct[source_pos] == ')' and struct[target_pos] == '('):

        if (seq[source_pos] == "A" and seq[target_pos] == "U") or (seq[source_pos] == "U" and seq[target_pos] == "A"):
            return "2H"

        elif (seq[source_pos] == "G" and seq[target_pos] == "C") or (seq[source_pos] == "C" and seq[target_pos] == "G"):
            return "3H"

        elif (seq[source_pos] == "G" and seq[target_pos] == "U") or (seq[source_pos] == "U" and seq[target_pos] == "G"):
            return "wobble"

    elif abs(source_pos - target_pos) == 1:
        return "phosphodiester"

def create_edge_index(sequence, structure):

  source_node = list(range(0, len(sequence) - 1))
  target_node = list(range(1, len(sequence)))

  if stemloop_check(structure):
    source_node.extend(pos_left_stem(structure))
    target_node.extend(pos_right_stem(structure))

  edge_index = [source_node, target_node]

  return torch.tensor(edge_index)

def create_node_attributes(sequence, start_pos, structure):

  nuc_to_onehot = {
      'A': [1, 0, 0, 0],
      'C': [0, 1, 0, 0],
      'G': [0, 0, 1, 0],
      'U': [0, 0, 0, 1]
  }

  node_attributes = [nuc_to_onehot[n] + [start_pos + i] + [round(float(compute_avg_delta_mfe(i, sequence, structure)), 3)] for i, n in enumerate(sequence)]
  return torch.tensor(node_attributes)

def create_edge_attributes(edge_index, sequence, structure):

  edge_to_onehot = {
      'phosphodiester': [1, 0, 0, 0],
      '2H': [0, 1, 0, 0],
      '3H': [0, 0, 1, 0],
      'wobble': [0, 0, 0, 1]
    }

  edge_attribute = []
  for i in range(len(edge_index[0])):
    edge_attribute.append(edge_to_onehot[type_of_bond(edge_index[0][i], edge_index[1][i], sequence, structure)])

  return torch.tensor(edge_attribute)

def create_graph_attributes(type, stemloop_len, stem_len, loop_len, gc_content, stemloop_id, tot_stemloops, start_pos, sequence_length, stemloop_mfe):

  tumorigenicity = {
      "non_cancer" : 0,
      "cancer" : 1
  }

  normal_stemloop = (int(stemloop_id.split("_")[-1].strip('slh')) + 1) / tot_stemloops
  normal_pos = start_pos / sequence_length

  graph_attribute = [tumorigenicity[type], stemloop_len, stem_len, loop_len, normal_stemloop, normal_pos, gc_content, stemloop_mfe]

  return torch.tensor(graph_attribute)

folder_path = "desired output folder path"

cancer_dataset = []
for idx in range(len(cdf)):
  entry = cdf.iloc[idx]
  symbol, sequence_length, tot_stemloops, stemloop_id, stemloop_seq, stemloop_struct, stemloop_len, stem_len, loop_len, start_pos, gc_content, stemloop_mfe = parse_entry(entry)
  print_loading_bar(idx, len(cdf), 50, f"Encoding {stemloop_id}...")
  edge_index = create_edge_index(stemloop_seq, stemloop_struct)
  node_attributes = create_node_attributes(stemloop_seq, start_pos, stemloop_struct)
  edge_attributes = create_edge_attributes(edge_index, stemloop_seq, stemloop_struct).float()
  data = Data(x=node_attributes, edge_index=edge_index, edge_attr=edge_attributes)
  data.graph_attributes = create_graph_attributes("cancer", stemloop_len, stem_len, loop_len, gc_content, stemloop_id, tot_stemloops, start_pos, sequence_length, stemloop_mfe)
  data.symbol = symbol
  data.id = stemloop_id
  data.cancer_association = True
  cancer_dataset.append(data)

torch.save(cancer_dataset, folder_path + "desired filenmae")
print_loading_bar(len(cdf), len(cdf), 50, "All cancer associated lncrna's stemloop modules encoded into graphs.")

folder_path = "desired output folder path"

non_cancer_dataset = []
for idx in range(len(ncdf)):
  entry = ncdf.iloc[idx]
  symbol, sequence_length, tot_stemloops, stemloop_id, stemloop_seq, stemloop_struct, stemloop_len, stem_len, loop_len, start_pos, gc_content, stemloop_mfe = parse_entry(entry)
  print_loading_bar(idx, len(ncdf), 50, f"Encoding {stemloop_id}...")
  edge_index = create_edge_index(stemloop_seq, stemloop_struct)
  node_attributes = create_node_attributes(stemloop_seq, start_pos, stemloop_struct)
  edge_attributes = create_edge_attributes(edge_index, stemloop_seq, stemloop_struct).float()
  data = Data(x=node_attributes, edge_index=edge_index, edge_attr=edge_attributes)
  data.graph_attributes = create_graph_attributes("non_cancer", stemloop_len, stem_len, loop_len, gc_content, stemloop_id, tot_stemloops, start_pos, sequence_length, stemloop_mfe)
  data.symbol = symbol
  data.id = stemloop_id
  data.cancer_association = False
  non_cancer_dataset.append(data)

torch.save(non_cancer_dataset, folder_path + "desired output filename")
print_loading_bar(len(ncdf), len(ncdf), 50, "All non-cancer associated lncrna's stemloop modules encoded into graphs.")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

graph_path = "desired graph path"

data = torch.load(graph_path, weights_only=False)

print("Graph summary:")
print(data)
print("-" * 150)

print("Edge index (connections between nodes):")
print(data.edge_index)
print("-" * 150)

if hasattr(data, 'x'):
    print("Node feature matrix (x):")
    print(data.x)
else:
    print("No node features found.")

print("-" * 150)

if hasattr(data, 'edge_attr'):
    print("Edge attributes:")
    print(data.edge_attr)
else:
    print("No edge attributes found.")

print("-" * 150)

print("Graph-level attributes:")
for key in data.keys():
    if key not in ['x', 'edge_index', 'edge_attr']:
        print(f"{key}: {getattr(data, key)}")

print("-" * 150)

G = nx.Graph()
edges = data.edge_index.t().tolist()
nts = ['A', 'C', 'G', 'U']

node_labels = {}
for i in range(data.num_nodes):
    if hasattr(data, 'x'):
        feat = data.x[i].tolist()
        label = nts[feat.index(1)] if 1 in feat else str(feat)
    else:
        label = str(i)
    node_labels[i] = label
    G.add_node(i, label=label)

edge_attr_map = {
    (1, 0, 0, 0): 'P',
    (0, 1, 0, 0): '2H',
    (0, 0, 1, 0): '3H',
    (0, 0, 0, 1): 'W'
}

backbone_edges, hbond_edges = [], []
edge_labels = {}

for i, (u, v) in enumerate(edges):
    attr_text = ""
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        vec = tuple(int(x) for x in data.edge_attr[i].tolist())
        attr_text = edge_attr_map.get(vec, str(vec))
    else:
        attr_text = ""

    edge_labels[(u, v)] = attr_text

    if attr_text in ['2H', '3H', 'W']:
        hbond_edges.append((u, v))
    else:
        backbone_edges.append((u, v))

paired = set([n for e in hbond_edges for n in e])
loop_nodes = [n for n in range(data.num_nodes) if n not in paired]

pos = {}

stem_pairs = hbond_edges
stem_height = 0
for u, v in stem_pairs:
    pos[u] = (-1, stem_height)
    pos[v] = (1, stem_height)
    stem_height += 1

if loop_nodes:
    theta = np.linspace(np.pi, 0, len(loop_nodes))
    radius = 1.5
    for i, n in enumerate(loop_nodes):
        pos[n] = (radius * np.cos(theta[i]), stem_height + radius * np.sin(theta[i]))

plt.figure(figsize=(7, 7))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

nx.draw_networkx_edges(G, pos, edgelist=backbone_edges, edge_color='black', width=1.5)
nx.draw_networkx_edges(G, pos, edgelist=hbond_edges, edge_color='red', style='--', width=2)

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkgreen', font_size=10)

plt.title("RNA Stem-Loop Secondary Structure")
plt.axis("off")
plt.show()