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

def comp_pars(filename):
    words = []
    with open('data/{}'.format(filename)) as f:
        for line in f.readlines():
            words.append([(s.lower(), None, is_cap(s), is_num(s)) for s in line.strip().split(' ')])
    return words

def results_row(words, tags):
    res = [words[i]+'_'+tags[i] for i in range(len(words))]
    return " ".join(res)

def write_results(filename, rows):
    with open("{}.wtag".format(filename), 'w') as f:
        for row in rows:
            print(row, file=f)

