import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
# Initialize ray to prevent user warning

def read_csv(filename, no_label=False, normalize=True, use_modin=True):
    if not use_modin:
        global pd
        import pandas as pd
    else:
        import modin.pandas as pd
        import ray
        ray.init()

    df = pd.read_csv(filename)
    tuples = df.loc[:, ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto']]
    tuples['pkts'] = df['orig_pkts'] + df['resp_pkts']
    tuples['start_ts'] = df['ts']
    tuples['last_ts'] = df['ts'] + df['duration']
    df = preprocess(df, no_label, normalize)
    print(df.columns.values)
    if no_label:
        return tuples, df
    else:
        Y = df['attack']
        X = df.drop(['attack'], axis=1)
        return X, Y

def preprocess(data, no_label, normalize):

    def normalize(df):
        scaler = MinMaxScaler()
        numeric = df[df.columns[~df.columns.isin(['attack'])]]
        df[df.columns[~df.columns.isin(['attack'])]] = scaler.fit_transform(numeric)
        return df
    

    bins = [1, 5, 10, 20, 30, 40, 50, 60, 120, -1]
    s_bins = [f'bin_{x}' for x in bins]
    selected_columns = ['request_body_len', 'response_body_len', 'trans_depth', 'is_http', 'established', 'is_ssl', 'proto', 'is_dns', 'is_conn', 'orig_pkt_rate', 'resp_pkt_rate', 'conn_state', 'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'sload', 'dload','orig_pkts', 'resp_pkts', 'smean', 'dmean', 'orig_ip_bytes', 'resp_ip_bytes', 'history', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_same_mthd'] + s_bins

    if not no_label:
        selected_columns.append('attack')
    data = data.loc[:, selected_columns]
    # HTTP body length ratio
    data.loc[:, 'http_len_ratio'] = data['request_body_len']/data['response_body_len']
    data.loc[:, ['conn_state', 'history']] = data.loc[:, ['conn_state', 'history']].replace(np.nan, '')

    # Calculate bins ratio
    bins_sum = data[[f'bin_{x}' for x in bins]].sum(axis=1)
    for b in bins:
        data.loc[:, f'bin_{b}'] /= bins_sum
    data.replace([np.nan, np.inf, 'F'], 0, inplace=True)
    data.replace('T', 1, inplace=True)

    # one-hot encode proto field. tcp, udp
    data.loc[:, 'proto_tcp'] = 0
    data.loc[:, 'proto_udp'] = 0
    data.loc[:, 'proto_icmp'] = 0
    data.loc[data['proto'] == 'tcp', f'proto_tcp'] = 1
    data.loc[data['proto'] == 'udp', f'proto_udp'] = 1
    data.loc[data['proto'] == 'icmp', f'proto_icmp'] = 1

    # one-hot encode conn_state field. S0, S1, SF, REJ, S2, S3, RSTO, RSTR, RSTOS0, RSTRH, SH, SHR, OTH
    fields = ['S0', 'S1', 'SF', 'REJ', 'S2', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'RSTRH', 'SH', 'SHR', 'OTH']
    for f in fields:
        data.loc[:, f'conn_state_{f}'] = 0
        data.loc[data['conn_state'] == f, f'conn_state_{f}'] = 1

    # one-hot encode history field. S, H, A, D, F, R
    fields = ['S', 'H', 'A', 'D', 'F', 'R']
    for f in fields:
        data.loc[:, f'history_{f}'] = 0
        data.loc[data['history'].str.contains(f, na=False), f'history_{f}'] = 1
    data.drop(['proto', 'conn_state', 'history'], axis=1, inplace=True)

    #print(data[[f'bin_{b}' for b in bins]])
    if normalize: data = normalize(data)
    return data

def print_result(test_Y, pred_Y):
    print('Accuracy: ', accuracy_score(test_Y, pred_Y))
    print('Recall: ', recall_score(test_Y, pred_Y))
    print('Precision: ', precision_score(test_Y, pred_Y))
    print('F1: ', f1_score(test_Y, pred_Y))
