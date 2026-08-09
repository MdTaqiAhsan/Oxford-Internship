import pandas as pd

FILES = [
    "cyto_2023_11_21_tcell_04.csv",
    "cyto_2023_11_21_ccell_04.csv"
]

for file in FILES:
    print(f"\nReading: {file}")

    total_rows = 0
    last_frame = None

    for chunk in pd.read_csv(
        file,
        usecols=["FRAME"],
        chunksize=100_000
    ):
        total_rows += len(chunk)

        # Track highest frame encountered
        chunk_max = chunk["FRAME"].max()

        if last_frame is None:
            last_frame = chunk_max
        else:
            last_frame = max(last_frame, chunk_max)

    print(f"Total data rows: {total_rows:,}")
    print(f"Last frame: {last_frame}")