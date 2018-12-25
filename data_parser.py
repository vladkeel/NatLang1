import numpy
import scipy

def is_cap(word):
    return word[0].isupper()

def is_num(word):
    return any(char.isdigit() for char in word)

def parse(filename):
    words = []
    with open('data/{}'.format(filename)) as f:
        for line in f.readlines():
            words.append([(s.split('_')[0].lower(), s.split('_')[1], is_cap(s.split('_')[0]), is_num(s.split('_')[0])) for s in line.strip().split(' ')])
    return words



