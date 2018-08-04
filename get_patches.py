#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import glob, os, csv, sys

dir_path = 'csvs/geo1/'
np.set_printoptions(threshold=np.nan)

SIZE_CUTOFF = 5

class Patch:
    def __init__(self, i, j):
        self.x = i
        self.y = j
        self.patch = None

def get_data(rel_path):
    abs_path = os.path.abspath(dir_path)
    os.chdir(abs_path)
    getdim = np.genfromtxt("2001.csv", delimiter=',')
    dim1 = len(getdim)
    dim2 = len(getdim[0])

    #maps = np.zeros((len(os.listdir(abs_path)),dim1,dim2))
    maps = []
    for i,file in enumerate(os.listdir(abs_path)):
        if file.endswith(".csv"):
            with open(file, 'rU') as p:
                year = file[0:4]
                m = np.genfromtxt(p, delimiter=',')
            maps.append( {'map': m, 'yr': int(year)})

    maps.sort(key=lambda x: x['yr'])
    return maps

def show_patch_map(patch_map, yr):
    im = np.zeros((len(patch_map), len(patch_map[0])))
    for i in range(len(patch_map)):
        for j in range(len(patch_map[0])):
            im[i][j] = patch_map[i][j].patch

    im = np.ma.masked_where(im == -1, im)

    cmap = plt.cm.rainbow
    cmap.set_bad(color='black')

    plt.imshow(im, cmap=cmap)
    plt.colorbar()
    plt.savefig('pngs/' + yr+'.png', bbox_inches='tight')
    plt.clf()
    #plt.show()

def write_patch_map(patch_map, yr):
    im = np.zeros((len(patch_map), len(patch_map[0])))
    for i in range(len(patch_map)):
        for j in range(len(patch_map[0])):
            im[i][j] = patch_map[i][j].patch
    np.savetxt("geo5_patch_maps/" + yr + ".csv", im, fmt='%d', delimiter=",")

ct = 0
def find_patches(map, yr):
    def on_map(i,j):
        x = len(map)
        y = len(map[0])
        if i >= x or i < 0 or j >= y or j < 0:
            return False
        return True

    patch_lattice = [[Patch(i,j) for j in range(0, len(map[0]))] for i in range(0,len(map))]
    patch_val = 1
    for i in range(len(patch_lattice)):
        for j in range(len(patch_lattice[0])):
            if map[i][j] == 0:
                patch_lattice[i][j].patch = -1
    for i in range(len(patch_lattice)):
        for j in range(len(patch_lattice[0])):
            if map[i][j] == 1 and patch_lattice[i][j].patch == None:
                #patch_lattice = mark_patch(patch_lattice, i,j, patch_val)
                q = [patch_lattice[i][j]]
                while len(q) > 0:
                    tmp = q.pop()
                    x = tmp.x
                    y = tmp.y
                    tmp.patch = patch_val
                    if on_map(x-1, y) and patch_lattice[x-1][y].patch == None:
                        q.append(patch_lattice[x-1][y])
                    if on_map(x+1, y) and patch_lattice[x+1][y].patch == None:
                        q.append(patch_lattice[x+1][y])
                    if on_map(x, y-1) and patch_lattice[x][y-1].patch == None:
                        q.append(patch_lattice[x][y-1])
                    if on_map(x, y+1) and patch_lattice[x][y+1].patch == None:
                        q.append(patch_lattice[x][y+1])
                patch_val += 1

    patches = [[] for i in range(0, 1+patch_val)]
    for i in range(len(patch_lattice)):
        for j in range(len(patch_lattice[0])):
            val = patch_lattice[i][j].patch
            patches[val].append(patch_lattice[i][j])

    patches = patches[0:len(patches)]
    # Remove patches that are 2px or smaller
    for patch_arr in patches:
        if len(patch_arr) < SIZE_CUTOFF:
            for patch_obj in patch_arr:
                patch_obj.patch = -1
            patch_arr = []


    patches_fixed = []
    for patch_arr in patches:
        if len(patch_arr) >= SIZE_CUTOFF:
            patches_fixed.append(patch_arr)
    patches_fixed.sort(key=lambda x: len(x), reverse=True)


    for i, patch_arr in enumerate(patches_fixed):
        for patch_obj in patch_arr:
            if (patch_obj.patch != -1):
                x = patch_obj.x
                y = patch_obj.y
                patch_lattice[x][y].patch = i

    n_patches = len(patches_fixed)
    for i in range(len(patch_lattice)):
        for j in range(len(patch_lattice[0])):
            if patch_lattice[i][j].patch > n_patches:
                print 'fuck'
    #write_patch_map(patch_lattice, yr)
    show_patch_map(patch_lattice, yr)


maps = get_data(dir_path)

for i,map in enumerate(maps):
    find_patches(map['map'], str(map['yr']))

#find_patches(maps[5]['map'], '2006')
