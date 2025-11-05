import os
import pandas as pd

folder_path = 'raw data folder path'

filenames = os.listdir(folder_path)

def get_symbols(filename):

  df = pd.read_csv(folder_path + filename)
  symbols = list(df["symbol"])
  return symbols

symbols = []
for filename in filenames:
  file_symbols = get_symbols(filename)
  symbols.extend(file_symbols)

symbols = list(set(symbols))
len(symbols)

output_path = 'desired output path'

with open(output_path, "w") as file:
  for symbol in symbols:
    file.write(f"{symbol}\n")