#!/usr/bin/env python3

from extract_lncrna_symbols import extract_lncrna_symbols
from retrieve_lncrna_sequence import retrieve_lncrna_sequence
from fasta_filehandling import fasta_filehandling
from secondary_structure_prediction import secondary_structure_prediction
from extract_stemloop_data import rna_structural_analysis, extract_stemloop_data
from stemloop_encoding import stemloop_encoding
from graph_embedding import graph_embedding
from clustering import clustering

DATASETS = [
    {   
        "type": "cancer",
        "base": "data/cancer-associated",
        "symbols": "cancer-associated-lncrna-symbols.txt",
        "fasta": "cancer-associated-lncrna.fasta",
        "excel": "cancer-associated-lncrna-sequences.xlsx",
        "structure": "cancer-associated-lncrna-structures.xlsx",
        "xml": "cancer-associated-lncrna-structural-analysis",
        "stemloop": "cancer-associated-lncrna-stemloops.xlsx",
        "graph": "cancer-associated-lncrna-graphs.pt"      
    },
    {
        "type": "non-cancer",
        "base": "data/non-cancer-associated",
        "symbols": "non-cancer-associated-lncrna-symbols.txt",
        "fasta": "non-cancer-associated-lncrna.fasta",
        "excel": "non-cancer-associated-lncrna-sequences.xlsx",
        "structure": "non-cancer-associated-lncrna-structures.xlsx",
        "xml": "non-cancer-associated-lncrna-structural-analysis",
        "stemloop": "non-cancer-associated-lncrna-stemloops.xlsx",
        "graph": "non-cancer-associated-lncrna-graphs.pt"    
    }
]

print("Processing both datasets to get stemloop data...")
for dataset in DATASETS:
    extract_lncrna_symbols(dataset["base"], dataset["symbols"])
    retrieve_lncrna_sequence(dataset["base"], dataset["symbols"], dataset["fasta"])
    fasta_filehandling(dataset["base"], dataset["fasta"], dataset["excel"])
    secondary_structure_prediction(dataset["base"], dataset["excel"], dataset["structure"])
    rna_structural_analysis(dataset["base"], dataset["structure"], dataset["xml"])
    extract_stemloop_data(dataset["base"], dataset["xml"], dataset["stemloop"])
    stemloop_encoding(dataset["base"], dataset["stemloop"], dataset["type"], dataset["graph"])

print("Graph Embedding...")
graph_embedding(
    graph_paths=[
            "data/cancer-associated/processed-data/cancer-associated-lncrna-graphs.pt",
            "data/non-cancer-associated/processed-data/non-cancer-associated-lncrna-graphs.pt"
       ],
    )

print("Clustering...")
clustering("data/results/graph_embedded_vectors.xlsx")