from scapy.all import *
import pandas as pd
import numpy as np
import sys

file_idx = int(sys.argv[1]) # Fraction index
mal_file = sys.argv[2] # Malicious record to map back

proto_map = {'tcp': TCP, 'udp': UDP, 'icmp': ICMP}
mal = pd.read_csv(mal_file)
mal.sort_values(['start_ts', 'last_ts'], inplace=True)

pkts = PcapReader(f'./split{file_idx}')
mal_pkts = []
for cur_p, pkt in enumerate(pkts):
    for idx, m in mal.iterrows():
        sts = m['start_ts'] - 1
        lts = m['last_ts'] + 1

        if (sts <= pkt.time and np.isnan(lts)) or \
            sts <= pkt.time <= lts:
            sip = m['id.orig_h']
            spt = m['id.orig_p']
            dip = m['id.resp_h']
            dpt = m['id.resp_p']
            proto = proto_map[m['proto']]
            if IPv6 in pkt: field = IPv6
            elif IP in pkt: field = IP
            else: continue

            if pkt[field].src == sip and pkt[field].dst == dip and proto in pkt and pkt[proto].sport == spt and pkt[proto].dport == dpt:
                mal_pkts.append(pkt)
                break
            elif pkt[field].src == dip and pkt[field].dst == sip and proto in pkt and pkt[proto].sport == dpt and pkt[proto].dport == spt:
                mal_pkts.append(pkt)
                break
        elif not np.isnan(lts) and pkt.time > lts:
            mal.drop([idx], inplace=True)
        elif pkt.time < sts: break
    print(f'{cur_p}, mal: {len(mal_pkts)}, ts: {sts}', end='\r')
if len(mal_pkts) != 0:
    wrpcap(f'./labeled-{file_idx}', mal_pkts)
