from utility_functions import run_rnafold, print_loading_bar
import pandas as pd

file_path = "lncrna sequence excel file path"

lncrna_df = pd.read_excel(file_path)
num_lncrna = len(lncrna_df)

for idx in range(num_lncrna):
    symbol = lncrna_df.at[idx, "Symbol"]
    sequence = lncrna_df.at[idx, "Sequence"]
    seq_len = lncrna_df.at[idx, "Sequence_length"]

    print_loading_bar(idx, num_lncrna, 50, f"Predicting Structure for {symbol} which has {seq_len} bp", "magenta")

    structure, mfe = run_rnafold(sequence)
    lncrna_df.at[idx, "Secondary_structure_representation"] = structure
    lncrna_df.at[idx, "MFE"] = mfe

print_loading_bar(num_lncrna, num_lncrna, 50, f"Structure Prediction Completed.", "magenta")

lncrna_df.to_excel(file_path, index=False)