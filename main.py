#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy, os, heapq, csv

from plotting import setup, show
from fractal_gen import fractal
from get_patches import find_patches
from scipy.spatial import distance
from cli import calc_division, calc_cohesion, calc_cli

BOARD_SIZE = 128
NUM_AXES = 4
N_POINTS = 10
BASE_PROB = .5

N_GEN = 50
H = 0.8

ALPHA = 0.2

data_path = './fractal_data.csv'

class Patch():
    def __init__(self, num):
        self.cells = []
        self.patch = num
        pass
    def add_cell(self, tuple):
        self.cells.append(tuple)

    def calc_centroid(self):
        x_sum = 0.0
        y_sum = 0.0
        n = len(self.cells)
        for (x,y) in self.cells:
            x_sum += x
            y_sum += y
        self.centroid = [float(x_sum)/float(n), float(y_sum)/float(n)]

def sort_patches(patch_map):
    N = len(patch_map)

    patches = []

    for i in range(N):
        for j in range(N):
            if patch_map[i][j].patch != -1:
                new = True
                for p in patches:
                    if p.patch == patch_map[i][j].patch:
                        new = False
                        p.add_cell((i,j))
                        break
                if new:
                    tmp = Patch(patch_map[i][j].patch)
                    tmp.add_cell((i,j))
                    patches.append(tmp)

    for patch in patches:
        patch.calc_centroid()
    return patches

def construct_dispersal_matrix(patches):
    def incidence_func(patch_i, patch_j):
        d_ij = float(distance.euclidean(patch_i.centroid, patch_j.centroid))
        return np.exp(-1*ALPHA*d_ij)

    N = len(patches)
    M = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i != j:
                M[i, j] = incidence_func(patches[i], patches[j])

    eig = np.linalg.eigvals(M)
    return np.amax(eig)

n_steps = 10
h_grad = [float(n)/float(n_steps) for n in range(1,n_steps)]
cov_grad = [(x, x+0.1) for x in np.arange(0.5,0.9,0.1)]

n_rep = 20
n_done = 0
n_treatments = len(cov_grad)*len(h_grad)*n_rep

print 'n_treatments: %d' % (n_treatments)

fields=['n_patches', 'H','habitat_loss','lambda_m', 'cohesion', 'cli']
with open(data_path, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

for cov_lo, cov_hi in cov_grad:
    for h_val in h_grad:
        for i in range(0, n_rep):
            maps, frac = fractal(cov_lo, cov_hi, h_val, BOARD_SIZE)
            if maps is not None:
                patch_map = find_patches(maps)
                patches = sort_patches(patch_map)
                if len(patches) > 1:
                    lambda_m = construct_dispersal_matrix(patches)
                    cohesion = calc_cohesion(maps, patches)
                    cli = calc_cli([x.cells for x in patches])

                    fields=[len(patches), h_val, frac, lambda_m, cohesion, cli]
                    with open(data_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)


                n_done += 1

                print float(n_done)/float(n_treatments)


#fig, ax = setup(BOARD_SIZE)
#show(fig, ax, maps, ani=False)
