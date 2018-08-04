#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import glob, os, csv, sys
from scipy.spatial import distance


def read_csv(dir_path, file):
    path = dir_path + file
    abs_path = os.path.abspath(path)
    mat = np.genfromtxt(abs_path, delimiter=',')
    return mat

def sort_into_patches(matrix):
    n_patches = int(np.amax(matrix))
    patches = [[] for x in range(n_patches+1)]

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            patch_val = int(matrix[i][j])
            if (patch_val >= 0):
                patches[patch_val].append((i,j))
    return patches

def calc_cli(patches):
    gyrate = 0

    g = [0.0 for x in range(len(patches))]
    a = [0 for x in range(len(patches))]
    a_total = 0
    for i,patch in enumerate(patches):
        a_total += len(patch)


    for i,patch in enumerate(patches):
        if len(patch) > 0:
            x_sum = 0
            y_sum = 0
            z = len(patch)
            for x,y in patch:
                x_sum += x
                y_sum += y
            x_centroid = float(x_sum)/float(z)
            y_centroid = float(y_sum)/float(z)

            centroid = (x_centroid, y_centroid)

            g_ij = 0
            for x,y in patch:
                h_ijr = distance.euclidean((x,y), centroid)
                g_ij += float(h_ijr)/float(z)
            g[i] = g_ij
            a[i] = z

    cli = 0
    for i, g_ij in enumerate(g):
        cli += g_ij * float(a[i])/float(a_total)
    return cli

def calc_division(matrix, patches):
    total_area = len(matrix) * len(matrix[0])

    s = 0
    for i,patch in enumerate(patches):
        if len(patch) > 0:
            s += (float(len(patch))/float(total_area))**2
    return (1.0 - s)*100

def get_per_frag(matrix):
    total_area = len(matrix) * len(matrix[0])

    ct = 0
    for x in range(0, len(matrix)):
        for y in range(0, len(matrix[0])):
            if (matrix[x][y] == -1):
                ct += 1
    return float(ct)/float(total_area)

file_names = [str(x) + '.csv' for x in range(2001,2014)]
dirs = ['geo' + str(i) for i in range(1,6)]

def show_cli():
    for dir in dirs:
        dir_path = 'data/' + dir +'/patch_map_csvs/'
        x = []
        y = []

        for file in file_names:
            matrix = read_csv(dir_path, file)
            per_frag = get_per_frag(matrix)
            patches = sort_into_patches(matrix)
            cli = calc_cli(patches)

            x.append(per_frag)
            y.append(cli)


        plt.scatter(x,y, label=dir)

    plt.legend()
    plt.show()

def show_division():
    for dir in dirs:
        dir_path = 'data/' + dir +'/patch_map_csvs/'
        x = []
        y = []

        for file in file_names:
            matrix = read_csv(dir_path, file)
            per_frag = get_per_frag(matrix)
            patches = sort_into_patches(matrix)
            div = calc_division(matrix, patches)

            x.append(per_frag)
            y.append(div)


        plt.scatter(x,y, label=dir)

    plt.legend()
    plt.show()


#show_division()
show_cli()
