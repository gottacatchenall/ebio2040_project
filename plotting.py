import matplotlib.pyplot as plt
import numpy as np

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
