from pathlib import Path
import pandas as pd

def extract_lncrna_symbols(base_dir, output_file):

    base = Path(base_dir)
    input_folder = base / "raw-data"
    output_file = base / "processed-data" / output_file

    symbols = set() # To avoid duplicates

    # Extracting symbols from each file
    for file in input_folder.iterdir():
        df = pd.read_csv(file)
        symbols.update(df["symbol"].dropna().unique())

    # Exporting symbols
    with open(output_file, "w") as f:
        for symbol in sorted(symbols):
            f.write(f"{symbol}\n")