#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy, os
#from plot_board import setup, show
import numpy as np
import heapq

folder = "markov4"

BOARD_SIZE = 128
NUM_AXES = 4
N_POINTS = 10
BASE_PROB = .5

N_GEN = 50

def setup(GRID_SIZE):
    global size
    size = GRID_SIZE
    figure = plt.figure(figsize=(7,7))
    ax = plt.gca()
    ax.grid(linestyle='-', linewidth=.5)

    # Minor ticks
    ax.set_xticks(np.arange(-.5, GRID_SIZE, 1));
    ax.set_yticks(np.arange(-.5, GRID_SIZE, 1));

    # Gridlines based on minor ticks
    ax.grid(color='black', linestyle='-', linewidth=.3)
    ax.set_xlim([-.5, GRID_SIZE-.5])
    ax.set_ylim([-.5, GRID_SIZE-.5])
    plt.tick_params(axis='both', which='both', left='off', labelleft='off', bottom='off', labelbottom='off')
    return figure, ax

def show(figure, ax, m, save=False, ani=False):
    if (ani):
        im = plt.imshow(m[0], interpolation='none', aspect='equal', cmap="gist_earth")
    else:
        im = plt.imshow(m, interpolation='none', aspect='equal', cmap="gist_earth")
    def update(i):
        ax.grid(color='black', linestyle='-', linewidth=.3)
        im.set_array(m[i])
        #time_text.set_text('Generation: ' + str(i))
        return im,
    if ani:
        ani = animation.FuncAnimation(figure, update, frames=[i for i in range(len(m))], interval=1, blit=True)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'),)
        if save:
            ani.save('map.mp4', dpi=300)
    #time_text = ax.text(0, size+1, 'Generation: 0', fontsize=12)
    plt.show()

def generate_fractal(cutoffs):
  def std_normal():
    val =  np.random.normal(0,1)
    return val
  def f4(delta, a, b, c, d):
      avg = ((a+b+c+d)/(4.0))
      val =  avg + delta * std_normal();
      return val

  def f3(delta, a, b, c):
      val = (float(a+b+c)/float(3)) + delta * std_normal()
      return val
  X = np.zeros((BOARD_SIZE+1,BOARD_SIZE+1))
  delta = 1.0
  N = BOARD_SIZE

  maxlevel = int(np.log2(BOARD_SIZE))

  H = 0.7

  D = N
  d = N/2

  # Init corner values
  X[0][0] = delta*std_normal()
  X[0][N] = delta*std_normal()
  X[N][0] = delta*std_normal()
  X[N][N] = delta*std_normal()

  for stage in range(0, maxlevel):
    delta = delta * np.power(0.5, (0.5*H))
    # interpolate and offset points
    for x in range(d, N-d+1, D):
      for y in range(d, N-d+1,D):
        X[x][y] = f4(delta, X[x+d][y+d], X[x+d][y-d], X[x-d][y+d],X[x-d][y-d])
        #print 'X[' + str(x) + '][' + str(y) + '] = ' + str(X[x][y])

    delta = delta * (pow(0.5, 0.5*H))

    # boundary grid points
    for x in range(d, N-d+1, D):
      X[x][0] = f3(delta, X[x+d][0], X[x-d][0], X[x][d])
      X[x][N] = f3(delta, X[x+d][N], X[x-d][N], X[x][N-d])
      X[0][x] = f3(delta, X[0][x+d], X[0][x-d], X[d][x])
      X[N][x] = f3(delta, X[N][x+d], X[N][x-d], X[N-d][x])

    # interpolate and offset interior grid points

    for x in range(d, N-d+1, D):
      for y in range(D, N-d+1,D):
        X[x][y] = f4(delta, X[x][y+d], X[x][y-d], X[x+d][y], X[x-d][y])


    for x in range(D,N-d+1,D):
      for y in range(d, N-d+1, D):
        X[x][y] = f4(delta, X[x][y+d], X[x][y-d], X[x+d][y], X[x-d][y])


    D = D/2;
    d = d/2;


  #print X

  m = np.zeros((BOARD_SIZE,BOARD_SIZE))

  for i in range(0, BOARD_SIZE):
    for j in range(0, BOARD_SIZE):
        for lo,hi in cutoffs:
            if (X[i][j] > lo and X[i][j] < hi):
                m[i][j] = 0
            else:
                m[i][j] = 1



  return m

def fractal(frac_lo, frac_hi):

    c = 0
    frac = 0

    #cutoffs = [(-0.1, 0.2), (.3,.4), (0.7, .9), (1.4,1.7), (-1.2,-0.9), (-.4,-.2)]
    cutoffs = [(-0.2, 0.3)]
    m = np.ones((BOARD_SIZE,BOARD_SIZE))


    loopct= 0
    while (frac < frac_lo or frac > frac_hi):
        c = 0
        if (frac < 0.8):
            mn = generate_fractal(cutoffs)
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if (mn[i,j] == 0):
                        m[i,j] = 0
        elif (frac > 0.9):
            mn = generate_fractal(cutoffs)
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if (mn[i,j] == 0):
                        m[i,j] = 1


        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if m[i,j] > 1:
                    m[i,j] = 1
                if m[i,j] == 0:
                    c += 1
        frac = float(c) / float(BOARD_SIZE**2)
        loopct += 1
        if (loopct > 10):
            print 'failed'
            break

    return m


class _Heap(object):
   def __init__(self, initial=None, key=lambda x:x):
       self.key = key
       if initial:
           self._data = [(key(item), item) for item in initial]
           heapq.heapify(self._data)
       else:
           self._data = []

   def push(self, item):
       heapq.heappush(self._data, (self.key(item), item))

   def pop(self):
       return heapq.heappop(self._data)[1]


def over_time(m):
    def on_board(i,j):
        if (i >= 0 and i < BOARD_SIZE and j >= 0 and j < BOARD_SIZE):
            return True
        return False
    def count_surroundings(tmp, i,j):
        ct = 0
        for x in range(i-1, i+2):
            for y in range(j-1, j+2):
                if (on_board(x,y)):
                    if tmp[x,y] == 1:
                        ct += 1
        return ct
    def count_frac(tmp):
        c = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if tmp[i,j] == 0:
                    c += 1
        return float(c) / float(BOARD_SIZE**2)
    base_prob = 0.01
    rev_prob = 0.005
    num_per_gen = int(float(BOARD_SIZE * BOARD_SIZE * 0.9)/float(N_GEN))

    '''
    heap = _Heap()
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if m[i,j] == 0:
                key = np.random.randint(0,1000)
                heap.push((key, (i,j)))
    '''

    #maps = np.ones((N_GEN, BOARD_SIZE, BOARD_SIZE))
    maps = [m]

    frac = count_frac(m)

    while(frac > .05):
        tmp = copy.deepcopy(maps[len(maps)-1])

        frac = count_frac(tmp)

        frac_co = frac - 0.02
        if frac_co < 0:
            frac_co = 0

        while (frac > frac_co ):
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if tmp[i,j] == 0:
                        scale = count_surroundings(tmp, i,j)
                        if scale > 0:
                            if (np.random.uniform(0,1) < 5*scale*base_prob):
                                tmp[i,j] = 1
                        #else:
                            #if (np.random.uniform(0,1) < base_prob):
                            #    tmp[i,j] = 1

            frac = count_frac(tmp)

        #for i in range(num_per_gen):
        #    if (len(heap._data) > 0):
        #        (key, (i,j)) = heap.pop()
        #        tmp[i,j] = 0
        maps.append(tmp)


    return maps


n_runs = 50

for i in range(n_runs):
    folder = "markov" + str(i)
    os.mkdir("csvs/"+ folder)
    m = fractal(0.8,0.9)
    maps = over_time(m)

    for i,map in enumerate(maps):
        t = str(i)
        np.savetxt("csvs/"+ folder +"/" + t + ".csv", map, fmt='%d', delimiter=",")



#fig, ax = setup(BOARD_SIZE)
#show(fig, ax, maps, ani=True)
