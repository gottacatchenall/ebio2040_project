#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import glob, os, csv, sys
from scipy.spatial import distance

file_names = [str(x) + '.csv' for x in range(2001,2014)]
dirs = ['geo' + str(i) for i in range(1,6)]

def read_csv(dir_path, file):
    path = dir_path + file
    abs_path = os.path.abspath(path)
    try:
        mat = np.genfromtxt(abs_path, delimiter=',')
        return mat
    except:
        return None

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
    for patch in patches:
        cell_list = patch.cells
        if len(cell_list) > 0:
            s += (float(len(cell_list))/float(total_area))**2
    return (1.0 - s)*100

def get_per_frag(matrix):
    total_area = len(matrix) * len(matrix[0])

    ct = 0
    for x in range(0, len(matrix)):
        for y in range(0, len(matrix[0])):
            if (matrix[x][y] == -1):
                ct += 1
    return float(ct)/float(total_area)



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

    model_dir_path = "models/csvs/" + model_folder + "/patch_csvs/"
    model_file_names = [str(i) + ".csv" for i in range(0, 1000)]

    x = []
    y = []

    for i,file in enumerate(model_file_names):
        matrix = read_csv(model_dir_path, file)
        if matrix is not None:
            per_frag = get_per_frag(matrix)
            patches = sort_into_patches(matrix)
            cli = calc_cli(patches)

            x.append(per_frag)
            y.append(cli)
    plt.scatter(x,y, label="model")
    plt.legend()
    plt.show()

def calc_cohesion(matrix, patches):
    def on_map(i,j):
        x = len(matrix)
        y = len(matrix[0])
        if i >= x or i < 0 or j >= y or j < 0:
            return False
        return True

    def on_perim(x,y):
        if on_map(x+1, y) and matrix[x+1,y] != matrix[x,y]:
            return True
        if on_map(x-1, y) and matrix[x-1,y] != matrix[x,y]:
            return True
        if on_map(x, y+1) and matrix[x,y+1] != matrix[x,y]:
            return True
        if on_map(x, y-1) and matrix[x,y-1] != matrix[x,y]:
            return True
        return False
    def count_perim(patch):
        p = 0
        for x,y in patch:
            if on_perim(x,y):
                p += 1
        return p

    A = len(matrix) * len(matrix[0])


    sum_p_ij = 0.0
    sum_p_ij_times_root_aij = 0.0

    for patch in patches:
        cell_list = patch.cells
        if len(cell_list) > 0:
            a_ij = len(cell_list)
            p_ij = count_perim(cell_list)

            sum_p_ij += p_ij
            sum_p_ij_times_root_aij += p_ij * np.sqrt(a_ij)
    return (1.0-(sum_p_ij/sum_p_ij_times_root_aij))/(1.0-(1.0/np.sqrt(A)))




def show_cohesion():

    model_dirs = ["data/markov" + str(n) + "/" for n in range(0,50)]

    model_x = []
    model_y = []
    for dir in model_dirs:
        with open(dir + "cohesion.csv", "rb") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                model_x.append(float(line[0]))
                model_y.append(float(line[1]))
    plt.scatter(model_x, model_y)

    for dir in dirs:
        dir_path = 'data/' + dir +'/patch_map_csvs/'
        x = []
        y = []

        for file in file_names:
            matrix = read_csv(dir_path, file)
            per_frag = get_per_frag(matrix)
            patches = sort_into_patches(matrix)
            cohension_index = calc_cohesion(matrix, patches)

            x.append(per_frag)
            y.append(cohension_index)


        plt.scatter(x,y, label=dir)

    '''
    model_dir_path = "models/csvs/markov" + str(markov_n) + "/patch_csvs/"
    model_file_names = [str(i) + ".csv" for i in range(0, 100)]


    x = []
    y = []

    for i,file in enumerate(model_file_names):
        matrix = read_csv(model_dir_path, file)
        if matrix is not None:
            per_frag = get_per_frag(matrix)
            patches = sort_into_patches(matrix)
            cohension_index = calc_cohesion(matrix, patches)

            x.append(per_frag)
            y.append(cohension_index)

            with open("data/markov" + str(markov_n) + "/cohesion.csv" , 'a') as csvfile:
                 writer = csv.writer(csvfile)
                 writer.writerow([per_frag,cohension_index])
    '''
    #plt.scatter(x,y, label="model")
    plt.legend()
    plt.show()


def show_division():
    model_dirs = ["data/markov" + str(n) + "/" for n in range(0,50)]

    model_x = []
    model_y = []
    for dir in model_dirs:
        with open(dir + "division.csv", "rb") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                model_x.append(float(line[0]))
                model_y.append(float(line[1]))
    plt.scatter(model_x, model_y)

    for dir in dirs:
        dir_path = 'data/' + dir +'/patch_map_csvs/'
        x = []
        y = []

        for i,file in enumerate(file_names):
            matrix = read_csv(dir_path, file)
            if matrix is not None:
                per_frag = get_per_frag(matrix)
                patches = sort_into_patches(matrix)
                div = calc_division(matrix, patches)

                x.append(per_frag)
                y.append(div)



        plt.scatter(x,y, label=dir)


    '''
    model_dir_path = "models/csvs/markov" + str(markov_n) + "/patch_csvs/"
    model_file_names = [str(i) + ".csv" for i in range(0, 100)]

    x = []
    y = []

    for i,file in enumerate(model_file_names):
        matrix = read_csv(model_dir_path, file)
        if matrix is not None:
            per_frag = get_per_frag(matrix)
            patches = sort_into_patches(matrix)
            div = calc_division(matrix, patches)

            x.append(per_frag)
            y.append(div)

            with open("data/markov" + str(markov_n) + "/division.csv" , 'a') as csvfile:
                 writer = csv.writer(csvfile)
                 writer.writerow([per_frag,div])
    '''
    #plt.scatter(x,y, label="model")

    plt.legend()
    plt.show()

#show_cohesion()
#show_division()

#show_cli()
