import numpy
import scipy


def parse(filename):
    words = []
    with open('data/{}'.format(filename)) as f:
        for line in f.readlines():
            words.append([s.split('_') for s in line.strip().split(' ')])
    return words



