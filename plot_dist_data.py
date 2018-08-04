#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import glob, os, csv, sys

data_path = 'data/geo2/avg_dist/'

def load_data():
    file_names = [str(x) + '.csv' for x in range(2001,2014)]
    data = []
    for file in file_names:
        path = data_path + file
        abs_path = os.path.abspath(path)
        with open(abs_path, 'rb') as csvfile:
            data_o = {'year': file[0:4], 'vals': [] }
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data_o['vals'].append(float(row[2]))
            data.append(data_o)
    return data

data = load_data()

yrs = []
slope = []

for o in data:
    max = int(np.amax(o['vals']))
    x = []
    y = []
    for i in range(1, max+1):
        ct = 0
        for val in o['vals']:
            if val < i:
                ct += 1
        x.append(i)
        y.append(ct)
    s, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    yrs.append(int(o['year']))
    slope.append(s)
    #plt.plot(x,y, label=o['year'])

plt.plot(yrs, slope)
plt.legend()
plt.show()
