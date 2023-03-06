#!/usr/bin/env python3
import sys
import pandas as pd

IOC_FILE = sys.argv[1]+'/IOC.txt'
df = pd.read_csv(sys.argv[1]+'/full_data.csv')
filters = []
cur_filter = ''
with open(IOC_FILE) as f:
    while True:
        line = f.readline()
        if len(line) == 0: break
        if line[:3] == '---':
            filters.append(cur_filter[:-2])
            cur_filter = ''
        else:
            key, value = line.strip().split(' = ')
            cur_filter += f'(df["{key}"] == {value}) & '

if cur_filter != '':
    filters.append(cur_filter[:-2])

df.loc[:, 'attack'] = 0
for f in filters:
    df.loc[eval(f), 'attack'] = 1
df.to_csv(sys.argv[1]+'/full_data.csv')
