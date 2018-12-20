import data_parser as prs
from math import exp, log
import numpy as np
import datetime
import logging
import functools
import time
import coloredlogs
from scipy.optimize import fmin_l_bfgs_b as minimize
from scipy.sparse import csr_matrix as sparse_mat
from scipy.sparse import hstack
import sys

logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)
LAMBDA = 3
iteration_number = 1

def progress_bar(progress, text):
    """
    Prints progress bar to console
    Args:
        progress: float in [0,1] representing progress in action
                    where 0 nothing done and 1 completed.
        text: Short string to add after progress bar.
    """
    if isinstance(progress, int):
        progress = float(progress)
    block = int(round(20*progress))
    progress_line = "\rCompleted: [{0}] {1:5.2f}% {2}.".format("#"*block + "-"*(20-block), progress*100, text)
    sys.stdout.write(progress_line)
    sys.stdout.flush()

def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.warning(f"Finished {func.__name__!r} in {str(datetime.timedelta(seconds=run_time))}")
        return value

    return wrapper_timer


class Model:
    def __init__(self, train_data):
        logger.info("Initializing model")
        self.iteration_number = 1
        self.set_of_tags = set()
        self.train_sentences = []
        self.train_tags = []
        self.key_to_int = {}
        self.int = 0
        tag_enum = 0
        self.tag_to_int = {}
        self.int_to_tag = {}
        self.word_map = {}
        self.pi = None
        self.pi_computed = None
        self.bp = None
        logger.info("Collecting features and tags")
        for sentence in train_data:
            words = [a[0] for a in sentence]
            self.train_sentences.append(words)
            tags = [a[1] for a in sentence]
            self.train_tags.append(tags)
            for tag in tags:
                if tag not in self.set_of_tags:
                    self.set_of_tags.add(tag)
                    self.tag_to_int[tag] = tag_enum
                    self.int_to_tag[tag_enum] = tag
                    tag_enum += 1
            for idx, _ in enumerate(sentence):
                self.feature_collector(words, tags, idx)
        if '*' not in self.set_of_tags:
            self.set_of_tags.add('*')
            self.tag_to_int['*'] = tag_enum
            self.int_to_tag[tag_enum] = '*'
            tag_enum += 1
        logger.info("Collected {} features and {} tags".format(self.int, tag_enum))
        self.data_features = sparse_mat((0, self.int))
        self.data_alt_features = []
        self.v = None
        self.sum_of_features = sparse_mat((1, self.int))
        word_enum = 0
        logger.info("Extracting features")
        for i in range(len(self.train_sentences)):
            for idx in range(len(self.train_sentences[i])):
                self.word_map[(i, idx)] = word_enum
                word_enum += 1
                self.data_features = hstack([self.data_features.transpose(), sparse_mat(self.feature_extractor(
                    self.train_sentences[i], self.train_tags[i][idx],
                    self.train_tags[i][idx - 1] if idx > 0 else '*',
                    self.train_tags[i][idx - 2] if idx > 1 else '*', idx)).transpose()], format='csr').transpose()
                self.sum_of_features += self.data_features[word_enum - 1]
                word_alt_features = [None] * len(self.set_of_tags)
                for tag_id, tag in self.int_to_tag.items():
                    word_alt_features[tag_id] = self.feature_extractor(self.train_sentences[i], tag,
                                                                       self.train_tags[i][idx - 1] if idx > 0 else '*',
                                                                       self.train_tags[i][idx - 2] if idx > 1 else '*',
                                                                       idx)
                self.data_alt_features.append(sparse_mat(np.array(word_alt_features)))
            progress_bar(i/len(self.train_sentences), "completed {} of {} sentences".format(i, len(self.train_sentences)))
        logger.info("Extracted features for {} words".format(word_enum))

    def calculate_probability(self, words, tag, last_tag, last2_tag, idx, mahane):
        mone = exp(self.v.dot(sparse_mat(self.feature_extractor(words, tag, last_tag, last2_tag, idx))))
        return mone / mahane

    def feature_collector(self, words, tags, idx):
        current_tag = tags[idx]
        if current_tag not in self.key_to_int:
            self.key_to_int[current_tag] = self.int
            self.int += 1
        key = '{}_{}'.format(words[idx], current_tag)
        if key not in self.key_to_int:
            self.key_to_int[key] = self.int
            self.int += 1
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'suffix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                if key not in self.key_to_int:
                    self.key_to_int[key] = self.int
                    self.int += 1
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'prefix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                if key not in self.key_to_int:
                    self.key_to_int[key] = self.int
                    self.int += 1

        if idx == 0:
            key = '{}_{}_{}'.format('*', '*', current_tag)
        elif idx == 1:
            key = '{}_{}_{}'.format('*', tags[idx - 1], current_tag)
        else:
            key = '{}_{}_{}'.format(tags[idx - 2], tags[idx - 1], current_tag)

        if key not in self.key_to_int:
            self.key_to_int[key] = self.int
            self.int += 1

        if idx == 0:
            key = '{}_{}'.format('*', current_tag)
        else:
            key = '{}_{}'.format(tags[idx - 1], current_tag)

        if key not in self.key_to_int:
            self.key_to_int[key] = self.int
            self.int += 1

        if idx > 0:
            key = 'prev_{}_{}'.format(words[idx - 1], current_tag)
            if key not in self.key_to_int:
                self.key_to_int[key] = self.int
                self.int += 1

        if idx < len(words) - 1:
            key = 'next_{}_{}'.format(words[idx + 1], current_tag)
            if key not in self.key_to_int:
                self.key_to_int[key] = self.int
                self.int += 1

    @timer
    def L(self, v):
        logger.info('run L(v)')
        v_s = sparse_mat(v)
        tmp1 = np.sum(self.data_features.dot(v_s.transpose()).toarray())
        tmp2 = np.sum([np.log(np.sum(np.exp(mat.dot(v_s.transpose()).toarray()))) for mat in self.data_alt_features])
        return -(tmp1 - tmp2 - ((LAMBDA / 2) * v_s.dot(v_s.transpose())[0, 0]))

    @timer
    def dLdv(self, v):
        logger.info("run dldv")
        v_s = sparse_mat(v)
        exp_count = np.sum([sparse_mat(np.exp(mat.dot(v_s.transpose()).toarray())
                                              /np.sum(np.exp(mat.dot(
            v_s.transpose()).toarray()))).transpose().dot(mat)
                          for mat in self.data_alt_features]).transpose()
        logger.critical("iteration number: {}".format(self.iteration_number))
        self.iteration_number += 1
        return -np.subtract(np.subtract(self.sum_of_features.toarray(), exp_count.toarray()[:,0]), (LAMBDA * v_s).transpose().toarray()[:,0])

    def feature_extractor(self, words, tag, last_t, last2_t, idx):
        new_ret = np.zeros(self.int)
        key = '{}_{}'.format(words[idx], tag)
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'suffix{}_{}_{}'.format(i, words[idx][-i:], tag)
                if key in self.key_to_int:
                    new_ret[self.key_to_int[key]] = 1

        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'prefix{}_{}_{}'.format(i, words[idx][-i:], tag)
                if key in self.key_to_int:
                    new_ret[self.key_to_int[key]] = 1

        if idx == 0:
            key = '{}_{}_{}'.format('*', '*', tag)
        elif idx == 1:
            key = '{}_{}_{}'.format('*', last_t, tag)
        else:
            key = '{}_{}_{}'.format(last2_t, last_t, tag)

        if key in self.key_to_int:
            new_ret[self.key_to_int[key]] = 1

        if idx == 0:
            key = '{}_{}'.format('*', tag)
        else:
            key = '{}_{}'.format(last_t, tag)

        if key in self.key_to_int:
            new_ret[self.key_to_int[key]] = 1
        if tag in self.key_to_int:
            new_ret[self.key_to_int[tag]] = 1

        if idx > 0:
            key = 'prev_{}_{}'.format(words[idx - 1], tag)
            if key in self.key_to_int:
                new_ret[self.key_to_int[key]] = 1

        if idx < len(words) - 1:
            key = 'next_{}_{}'.format(words[idx + 1], tag)
            if key in self.key_to_int:
                new_ret[self.key_to_int[key]] = 1
        return new_ret

    def train(self):
        logger.debug('Start Now!!')
        self.v, f, d = minimize(self.L, np.zeros(self.int), factr=1e12, pgtol=1e-3, fprime=self.dLdv)
        logger.debug('End Now!!')
        logger.debug("v is: {}".format(self.v))
        logger.debug("Result of minimize is {}".format("success" if d['warnflag'] == 0 else "failure"))
        logger.debug("Function called {} times".format(d['funcalls']))
        logger.debug("Number of iterations {}".format(d['nit']))
        np.save('v', self.v)

    def pi_comp(self, k, u, v, words):
        if self.pi_computed[k, u, v] != 1:
            self.pi_computed[k, u, v] = 1
            pi_for_t = np.zeros(len(self.set_of_tags))
            for t in self.set_of_tags:
                pi_for_t[self.tag_to_int[t]] = \
                    self.pi_comp(k-1, self.tag_to_int[t], u)*\
                    self.calculate_probability(words, self.int_to_tag[v],self.int_to_tag[u], t, k-1)
            self.pi[k,u,v] = np.max(pi_for_t)
            self.bp[k,u,v] = np.argmax(pi_for_t)
        return self.pi[k, u, v]

    def infer(self, words):
        logger.debug('Infering for given sentence')
        self.pi = np.zeros((len(words) + 1, len(self.set_of_tags), len(self.set_of_tags)))
        self.pi_computed = np.zeros((len(words) + 1, len(self.set_of_tags), len(self.set_of_tags)))
        self.bp = np.zeros((len(words) + 1, len(self.set_of_tags), len(self.set_of_tags)))
        self.pi[0, self.tag_to_int['*'], self.tag_to_int['*']] = 1
        self.pi_computed[0, self.tag_to_int['*'], self.tag_to_int['*']] = 1
        for k in range(1, len(words)+1):
            for u in self.set_of_tags:
                for v in self.set_of_tags:
                    self.pi[k, self.tag_to_int[u], self.tag_to_int[v]] = \
                        self.pi_comp(k, self.tag_to_int[u], self.tag_to_int[v])
        tags = [None] * len(words)
        tags[len(tags)-1] = self.int_to_tag[np.argmax(self.pi[len(tags)])[1]]
        tags[len(tags)-2] = self.int_to_tag[np.argmax(self.pi[len(tags)])[0]]
        for k in range(len(words) - 3, -1, -1):
            tags[k] = self.int_to_tag[self.bp[k+3, self.tag_to_int[tags[k+1]], self.tag_to_int[tags[k+2]]]]
        return tags



if __name__ == '__main__':
    x = np.load('v.npy')
    all_sentences = prs.parse('train.wtag')
    mymodel = Model(all_sentences).train()
