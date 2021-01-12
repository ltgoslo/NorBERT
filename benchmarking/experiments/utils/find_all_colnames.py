import glob
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    files = glob.glob(r"../*/**/[a-zA-Z]*.xlsx", recursive=True)

    paths = []
    cols = []

    for file in files:
        df = pd.read_excel(file)
        paths.extend([file] * len(df.columns))
        cols.extend(df.columns)

    pd.DataFrame({"path": paths, "colname": cols}).to_csv(Path(__file__).parent / "all_colnames.tsv", index=False, sep="\t")
