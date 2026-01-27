import struct
import pandas as pd

df = pd.read_csv('graph_data.csv', header=None, names=['src', 'dest', 'offset'])

# Export to the binary format LearnedBench expects
# RSMI/LearnedBench usually expects: 
# [Double X][Double Y] ... for each point
with open('graph_data.bin', 'wb') as f:
    for _, row in df.iterrows():
        # Pack as two 64-bit doubles
        f.write(struct.pack('dd', float(row['src']), float(row['dest'])))
