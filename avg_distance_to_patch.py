#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import glob, os, csv, sys
from scipy.spatial import distance


dir_path = 'data/geo5/patch_map_csvs/'
data_path = 'data/geo5/avg_dist/'

def read_csv(file):
    path = dir_path + file
    abs_path = os.path.abspath(path)
    mat = np.genfromtxt(abs_path, delimiter=',')
    return mat

def sort_into_patches(matrix):
    def on_map(i,j):
        x = len(matrix)
        y = len(matrix[0])
        if i >= x or i < 0 or j >= y or j < 0:
            return False
        return True
    def is_surrounded(i,j, patch_val):
        surrounded = True
        if on_map(i-1, j) and matrix[i-1][j] != patch_val:
            surrounded = False
        if on_map(i+1, j) and matrix[i+1][j] != patch_val:
            surrounded = False
        if on_map(i, j-1) and matrix[i][j-1] != patch_val:
            surrounded = False
        if on_map(i, j+1) and matrix[i][j+1] != patch_val:
            surrounded = False

        return surrounded

    n_patches = int(np.amax(matrix))
    patches = [[] for x in range(n_patches+1)]

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            patch_val = int(matrix[i][j])
            if (patch_val >= 0):
                if (not is_surrounded(i,j, patch_val)):
                    patches[patch_val].append((i,j))
    return patches

def show_edges(matrix, patches):
    mat = np.zeros_like(matrix)
    for patch_arr in patches:
        for x,y in patch_arr:
            mat[x][y] = 1
    plt.imshow(mat)
    plt.show()


def get_min_dist_between_each_pair(patches, file_name):
    path = data_path + file_name
    abs_path = os.path.abspath(path)

    for i in range(1, len(patches)-1):
        for j in range(i+1, len(patches)):
            min = 10000
            for a in patches[i]:
                for b in patches[j]:
                    val = distance.euclidean(a, b)
                    if val < min:
                        min = val

            with open(abs_path, 'a') as csvfile:
                 writer = csv.writer(csvfile)
                 writer.writerow([i,j,min])

file_names = [str(x) + '.csv' for x in range(2001,2014)]

for file in file_names:
    matrix = read_csv(file)
    patches = sort_into_patches(matrix)
    get_min_dist_between_each_pair(patches, file)
