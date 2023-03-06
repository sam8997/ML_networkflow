#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
M = 100 # Statistical feature window
N = 60  # Fragemented network flow window
X = 60  # HTTP body ratio window

directory = sys.argv[1]
frag_idx = int(sys.argv[2])
data = pd.read_csv(f'{directory}/split_{frag_idx:02d}.csv')
prev_datalen = 0
if frag_idx > 0:
    data_prev = pd.read_csv(f'{directory}/split_{(frag_idx-1):02d}.csv')
    data_prev_N = data_prev[data_prev['ts'] >= data.loc[0, 'ts']-N]
    if data_prev_N.shape[0] < M: data_prev = data_prev.iloc[-M-1:]
    else: data_prev = data_prev_N

    prev_datalen = data_prev.shape[0]
    data = data_prev.append(data, ignore_index=True)
print(f"prev_datalen: {prev_datalen}")

print("Splitting data in different ts groups: ")
# Split data in different ts groups
cur_ts = data.loc[0,'ts']
prev_ts = 0
prev_cntr = 0
cur_cntr = 0
for idx, d in data.iterrows():
    if cur_ts != d['ts']:
        prev_cntr = cur_cntr
        cur_cntr = 0
        cur_ts = d['ts']
        prev_ts = cur_ts
    data.loc[idx, 'prev_off'] = prev_cntr + cur_cntr
    data.loc[idx, 'prev_ts'] = prev_ts
    data.loc[idx, 'group_size'] = data.loc[data['ts'] == d['ts']].shape[0]
    data.loc[idx, 'group_off'] = cur_cntr
    cur_cntr += 1
    print(f'{idx}/{data.shape[0]}', end='\r')

print("Calculating statistical features")
# Calculate statistical feature based on M connections
for idx, d in data.iterrows():
    # Calculate interval left/right
    d_right = idx + d['group_size'] - d['group_off']
    d_left = idx - d['group_off']

    if d_right - d_left < M: # If too few connection, add more connection
        d_left = idx - d['prev_off']
        pivot_d = data.loc[d_left]
        while d_right - d_left < M:
            if d_left == 0: break
            d_left = d_left - pivot_d['prev_off']
            pivot_d = data.loc[d_left]

    sample_df = data.loc[int(d_left):int(d_right)+1]
    sample_resp_h = sample_df[sample_df['id.resp_h'] == d['id.resp_h']]
    sample_orig_h = sample_df[sample_df['id.orig_h'] == d['id.orig_h']]

    if d['is_http'] == 1:
        data.loc[idx, 'ct_same_mthd'] = sample_orig_h.loc[data['method'] == d['method']].shape[0]

    data.loc[idx, 'ct_dst_ltm'] = sample_resp_h.shape[0]
    data.loc[idx, 'ct_src_ltm'] = sample_orig_h.shape[0]
    data.loc[idx, 'ct_src_dport_ltm'] = sample_orig_h.loc[(sample_df['id.resp_p'] == d['id.resp_p'])].shape[0]
    data.loc[idx, 'ct_dst_sport_ltm'] = sample_resp_h.loc[(sample_df['id.orig_p'] == d['id.orig_p'])].shape[0]
    data.loc[idx, 'ct_dst_src_ltm'] = sample_resp_h.loc[(sample_df['id.orig_h'] == d['id.orig_h'])].shape[0]
    print(f'{idx}/{data.shape[0]}', end='\r')

print("Calculating body ratio for http connections")
# Calculate body ratio for http connections based on X second window
for idx, d in data.iterrows():
    ts_right = d['ts']
    ts_left = d['ts'] - X
    size = data.loc[(data['ts'] < ts_right) &
                    (data['ts'] >= ts_left) &
                    (data['id.resp_h'] == d['id.resp_h']) &
                    (data['method'] == d['method']) &
                    (data['method'] != 0)].shape[0]
    data.loc[idx, 'ct_same_mthd'] = size
    print(f'{idx}/{data.shape[0]}', end='\r')

print("Merging fragmented network flow")
# Merge fragmented network flow that appears in N seconds (Same 5-tuple)
data.loc[:, 'frag_cntr'] = 0
for idx, d in data.iterrows():
    ts_right = d['ts']
    ts_left = d['ts'] - N
    sample_common = data.loc[(data['ts'] < ts_right) &
                             (data['ts'] >= ts_left) &
                             (data['id.orig_h'] == d['id.orig_h']) &
                             (data['id.resp_h'] == d['id.resp_h']) &
                             (data['proto'] == d['proto'])]

    x = sample_common.loc[(data['id.orig_p'] == d['id.orig_p'])]
    if x.shape[0] > 0:
        x = x.iloc[-1] # Get the nearest connection
        data.loc[idx, 'frag_cntr'] = x['frag_cntr'] + 1

    x = sample_common.loc[(data['id.resp_p'] == d['id.resp_p'])]
    if x.shape[0] > 0:
        x = x.iloc[-1] # Get the nearest connection
        data.loc[idx, 'frag_cntr'] = x['frag_cntr'] + 1
    print(f'{idx}/{data.shape[0]}', end='\r')

print("Calculating perioidic bins")
# Calculate periodic bins (1, 5, 10, 20, 30, 40, 50, 60)
bins = [1, 5, 10, 20, 30, 40, 50, 60, 120, -1]
data.loc[:, [f'bin_{x}' for x in bins]] = 0
grouped = data.groupby(['id.resp_h', 'id.resp_p'])
for k,_ in grouped:
    cur_group = grouped.get_group(k)
    for i in range(cur_group.shape[0]-1):
        off = cur_group.iloc[i+1]['ts']-cur_group.iloc[i]['ts']
        idx_1 = cur_group.iloc[i+1].name
        idx_0 = cur_group.iloc[i].name
        suc=False
        for b in bins[:-1]:
            if b > off:
                data.loc[idx_1, f'bin_{b}'] = data.loc[idx_0, f'bin_{b}']+1
                break
        if not suc:
            data.loc[idx_1, f'bin_-1'] += 1

data = data.iloc[prev_datalen:]
data.to_csv(f'{directory}/ct_{frag_idx:02d}.csv', index=False)
