#! /usr/bin/env python

import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

folder = 'geometry1_final_region/pngs/'
file_names = [str(x) + '.png' for x in range(2001,2014)]
print file_names
#file_names = ['2001.png']

ims = []

for file in file_names:
    im = imageio.imread(folder + file)
    ims.append(im)

matrix = np.zeros((len(ims), len(ims[0]), len(ims[0][0])), dtype=int)

for im in range(len(ims)):
    for i in range(0, len(ims[im])):
        for j in range(0, len(ims[im][0])):
            if ims[im][i][j][0] == 28:
                matrix[im][i][j] = 1
            #matrix[im][i][j] = norm_val

for i,m in enumerate(matrix):
    yr = str(i + 2001)
    np.savetxt("csvs/geo1/" + yr + ".csv", m, fmt='%d', delimiter=",")



'''
def show(m, save=False, ani=False):
    fig, ax = plt.subplots()
    im = ax.imshow(m[0], interpolation='none', aspect='equal', cmap="gist_earth")

    def update(i):
        #ax.grid(color='black', linestyle='-', linewidth=.3)
        im.set_array(m[i])
        #time_text.set_text('yr: ' + str(i))
        return im,

    ani = animation.FuncAnimation(fig, update, frames=[i for i in range(len(m))], interval=250, blit=False)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'),)
        ani.save('geometry5.mp4', dpi=300)
    #time_text = ax.text(0, 105, 'yr: 0', fontsize=12)
    plt.show()

show(matrix, ani=True, save=True)
'''
