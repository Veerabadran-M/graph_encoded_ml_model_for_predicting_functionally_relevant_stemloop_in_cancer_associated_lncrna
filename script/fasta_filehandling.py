import pandas as pd
from pathlib import Path
import os

def fasta_filehandling(base_dir, fasta_file, excel_file):

    base = Path(base_dir)
    file_path = base / "processed-data" / fasta_file

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

    raw_folder = base / "raw-data"

    raw_df = pd.DataFrame()
    for raw_file in raw_folder.iterdir():
        file = pd.read_csv(raw_file)
        file_df = pd.DataFrame(file)
        raw_df = pd.concat([raw_df, file_df])

    raw_df = raw_df.drop_duplicates(subset = ["symbol"])

    df = pd.merge(df, raw_df[['symbol', 'expression_detail']],
                    left_on='Symbol', right_on='symbol', how='left')
    df.drop(columns=['symbol'], inplace=True)
    df.rename(columns={'expression_detail': 'Expression'}, inplace=True)

    df["Sequence_length"] = df['Sequence'].apply(len)

    df = df[['Accession_ID', 'Symbol', 'Gene_ID', 'Expression', 'Sequence', 'Sequence_length']]

    output_filename = base / "processed-data" / excel_file
    df.to_excel(output_filename, index=False)