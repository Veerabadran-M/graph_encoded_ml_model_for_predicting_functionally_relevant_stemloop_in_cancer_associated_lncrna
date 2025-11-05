import sys

def print_loading_bar(current, total, bar_length=30, message="", color="cyan", line_width=120):
    
    COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "reset": "\033[0m",
}
    
    percentage = current / total * 100
    filled_length = int(bar_length * current // total)

    bar = COLORS.get(color, COLORS["reset"]) + \
          '█' * filled_length + '-' * (bar_length - filled_length) + \
          COLORS["reset"]

    line = f"[{bar}] {percentage:6.2f}% {message}"
    sys.stdout.write('\r' + line.ljust(line_width))
    sys.stdout.flush()

    if current == total:
        print()

def run_rnafold(seq: str):
    import RNA

    seq = seq.upper().replace("T", "U")
    structure, mfe = RNA.fold(seq)
    return structure, mfe

def compute_mfe(sequence, structure):
    import RNA

    fc = RNA.fold_compound(sequence)
    mfe = fc.eval_structure(structure)
    return mfe