#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os

def read_csv(filename):
    # Remove unused lines first
    modified_file = f'{filename}.mod'
    new_data = []
    with open(filename, 'r') as f:
        data = f.read().split('\n')[:-1]
        for d in data:
            if d[0] != '#':
                new_data.append(d)
            elif '#fields' in d:
                new_data.append(d[8:])
    with open(modified_file, 'w') as f:
        f.write('\n'.join(new_data))

    df = pd.read_csv(modified_file, sep='\t', low_memory=False)
    df = df.astype({'ts': np.int64})
    return df

base = sys.argv[1]
output_df = pd.DataFrame()

## Process http logs
print("Processing http logs ... ", end='')
filename = f'{base}/http.log'
if os.path.isfile(filename):
    df = read_csv(filename)
    df['is_http'] = 1
    output_df = output_df.append(df[['ts', 'uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','method','request_body_len','response_body_len','trans_depth', 'is_http']], ignore_index=True)
    print('done')
else:
    output_df[['method', 'request_body_len', 'response_body_len', 'trans_depth', 'is_http']] = 0
    print('not found')

# Process conn logs
print("Processing conn logs ... ", end='')
filename = f'{base}/conn.log'
if os.path.isfile(filename):
    df = read_csv(filename)
    df.replace({'-':'0'}, inplace=True)
    df = df.astype({'duration': np.float32, 'orig_bytes': np.int64, 'resp_bytes': np.int64})
    df.loc[df['service'] == 'dns', 'is_dns'] = 1                                                              
    df.loc[df['service'] == 'http', 'is_http'] = 1                                                            
    df.loc[df['service'] == 'ssl', 'is_ssl'] = 1                                                              
    df.loc[(df['is_dns'] == 0) & (df['is_ssl'] == 0) & (df['is_http'] == 0), 'is_conn'] = 1                   
    df['orig_pkt_rate'] = df['orig_pkts']/df['duration']
    df['resp_pkt_rate'] = df['resp_pkts']/df['duration']
    df['sload'] = df['orig_bytes']/df['duration']
    df['dload'] = df['resp_bytes']/df['duration']
    df['smean'] = df['orig_bytes']/df['orig_pkts']
    df['dmean'] = df['resp_bytes']/df['resp_pkts']

    output_df = output_df.append(
            df[['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p', 'proto', 'is_conn', 'is_dns', 'is_http', 'is_ssl',
                'orig_pkt_rate', 'resp_pkt_rate', 'conn_state', 'duration', 'orig_bytes', 'resp_bytes',
                'missed_bytes', 'sload','dload','orig_pkts','resp_pkts','smean','dmean',
                'orig_ip_bytes', 'resp_ip_bytes', 'history' ]], ignore_index=True)
    print('done')
else:
    output_df[['proto', 'is_conn', 'is_dns', 'is_http', 'is_ssl', 'orig_pkt_rate', 'resp_pkt_rate', 'conn_state', 
               'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'sload',
               'dload','orig_pkts','resp_pkts','smean','dmean', 'orig_ip_bytes',
               'resp_ip_bytes', 'history' ]] = 0
    print('not found')

# Map, remove .local dns query
print("Mapping dns logs ...", end='')
filename = f'{base}/dns.log'
if os.path.isfile(filename):
    df = read_csv(filename)
    df['is_dns'] = 1
    output_df = output_df.append(df[['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','trans_id', 'is_dns']])
    print('done')
else:
    print('not found')


# Map ssl logs
print("Mapping ssl logs ... ", end='')
filename = f'{base}/ssl.log'
if os.path.isfile(filename):
    df = read_csv(filename)
    for uid,es in zip(df['uid'], df['established']):
        output_df.loc[output_df['uid'] == uid, 'established'] = es
    print('done')
else:
    output_df['established'] = 0
    print('not found')

# Sort by ts
print('Sorting records by ts ... ', end='')
output_df = output_df.sort_values(by=['ts'])
output_df.fillna(0)
print('done')

# Save file
output_df.to_csv(f'{base}/data.csv', index=False)
print(f'File saved to {base}/data.csv')
