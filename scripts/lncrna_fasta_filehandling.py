import pandas as pd
import os

folder_path = 'dataset folder path'
file_path = os.path.join(folder_path, "fasta filename")

with open(file_path, 'r') as file:
    content = file.read()

entries = content.split(">")
entries.remove("")

tokens = [entry.split(" ") for entry in entries]

lncrna_accession_ids = [token[0] for token in tokens]
lncrna_symbols = [token[1] for token in tokens]
lncrna_ids = [token[4] for token in tokens]
lncrna_ids = [("".join(lncrna_id.split("]")[0])).split("=")[1] for lncrna_id in lncrna_ids]
sequences = [token[-1] for token in tokens]

lncrna_sequences = []
for seq in sequences:
    if seq[0] == "[":
        lncrna_sequences.append("".join(seq.split("\n")[1:]))
    else:
        lncrna_sequences.append(seq)

data = {'Accession_ID': lncrna_accession_ids, "Symbol" : lncrna_symbols, "Gene_ID" : lncrna_ids, "Sequence" : lncrna_sequences}
df = pd.DataFrame(data)

raw_folder_path = 'raw data folder path'

import os
raw_file_paths = os.listdir(raw_folder_path)

raw_df = pd.DataFrame()
for raw_file in raw_file_paths:
   file = pd.read_csv(raw_folder_path + raw_file)
   file_df = pd.DataFrame(file)
   raw_df = pd.concat([raw_df, file_df])

raw_df = raw_df.drop_duplicates(subset = ["symbol"])

df = pd.merge(df, raw_df[['symbol', 'expression_detail']],
                  left_on='Symbol', right_on='symbol', how='left')
df.drop(columns=['symbol'], inplace=True)
df.rename(columns={'expression_detail': 'Expression'}, inplace=True)

df["Sequence_length"] = df['Sequence'].apply(len)

df = df[['Accession_ID', 'Symbol', 'Gene_ID', 'Expression', 'Sequence', 'Sequence_length']]

output_filename = folder_path + 'desired output filename'
df.to_excel(output_filename, index=False)