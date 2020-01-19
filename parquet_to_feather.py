import pandas as pd
from pyarrow.feather import write_feather
import os
from pathlib import Path

INPUT_DIR = Path('parquet')
OUTPUT_DIR = Path('feather')

for f in os.listdir(INPUT_DIR):
    print(f'Reading {f}')
    df = pd.read_parquet(INPUT_DIR/f)
    fileout = f.replace('parquet','feather')
    print(f'Writing {fileout}')
    write_feather(df, OUTPUT_DIR/f.replace('parquet','feather'))
    
