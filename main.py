from sys import argv

import librosa
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == "__main__":
    y, sr = librosa.load(argv[1])
    y = np.abs(y)
    x = np.arange(len(y)) / sr
    print('y.shape:', y.shape)
    print('sample_rate:', sr)

    t_start = int(argv[2])
    t_end = int(argv[3])

    i_start = t_start * sr
    i_end = t_end * sr if t_end > 0 else -1

    y = y[i_start: i_end]
    x = x[i_start: i_end]

    MOV_N = 2
    avg_y = moving_average(y, n=MOV_N)
    avg_x = x[: -MOV_N + 1]

    # plt.plot(avg_x, avg_y, 'bo')
    # plt.show()
    THRESHOLD = 0.4
    DURATION = 0.01
    ret = []
    for i in range(1, len(y)):
        if y[i - 1] < THRESHOLD and y[i] >= THRESHOLD:
            if len(ret) == 0 or (len(ret) > 0 and x[i] - ret[-1][0] > DURATION):
                ret.append((x[i], y[i]))

    print('num pops:', len(ret))
    pprint(ret[:10])

    xs = [p[0] for p in ret]
    plt.hist(xs, bins=20)
    plt.show()