
import os, csv
import math

import numpy as np 

def read_logs(file_path):
    logs = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            logs.append(row)
    return logs

def process_logs(logs):
    processed = []
    # data: t, bat, h, tof, yaw, vgx, vgy, vgz
    for entry in logs:
        processed_entry = {
            'time': float(entry['t']),
            'battery': int(entry['bat']),
            'height': int(entry['h']),
            'time_of_flight': int(entry['tof']),
            'pitch': int(entry['pitch']),
            'roll': int(entry['roll']),
            'yaw': int(entry['yaw']),
            'vx': int(entry['vgx']),
            'vy': int(entry['vgy']),
            'vz': int(entry['vgz']),
        }
        processed.append(processed_entry)
    
    # process time to be relative and in seconds
    t0 = processed[0]['time']
    for i in range(1, len(processed)):
        time = processed[i]['time'] - t0
        processed[i]['time'] = time

    return processed

