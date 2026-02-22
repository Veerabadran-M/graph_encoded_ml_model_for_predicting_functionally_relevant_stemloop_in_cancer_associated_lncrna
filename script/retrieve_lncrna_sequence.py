import subprocess
import zipfile
import shutil
import os
from os.path import join
from pathlib import Path


def retrieve_lncrna_sequence(base_dir, symbol_file, output_fasta):

    base = Path(base_dir)
    zip_path = base / "output.zip"
    inp_file = base / "processed-data" / symbol_file

    # Download lncrna sequences using NCBI CL tool datasets
    subprocess.run(
        [
            "datasets",
            "download",
            "gene",
            "symbol",
            "--inputfile", str(inp_file),
            "--filename", str(zip_path)
        ],
        check=True
    )

    # Extracting the output zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base)

    # Moving and renaming the lncrna sequence file
    src = base / "ncbi_dataset/data/rna.fna"
    dst = base / "processed-data" / output_fasta
    shutil.move(src, dst)

    # Delete unwanted files and directories
    for file in ["md5sum.txt", "README.md", "output.zip"]:
        file_path = base / file
        os.remove(file_path)
    ncbi_dir = base / "ncbi_dataset"
    shutil.rmtree(ncbi_dir)