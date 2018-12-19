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

logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)
LAMBDA = 3
iteration_number = 1


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
        self.data_features = []
        self.data_alt_features = []
        self.v = None
        self.sum_of_features = np.zeros(self.int)
        word_enum = 0
        for i in range(len(self.train_sentences)):
            for idx in range(len(self.train_sentences[i])):
                self.word_map[(i, idx)] = word_enum
                word_enum += 1
                self.data_features.append(self.feature_extractor(self.train_sentences[i], self.train_tags[i][idx],
                                                                self.train_tags[i][idx - 1] if idx > 0 else '*',
                                                                self.train_tags[i][idx - 2] if idx > 1 else '*', idx))
                self.sum_of_features += self.data_features[word_enum - 1]
                word_alt_features = [None] * len(self.set_of_tags)
                for tag_id, tag in self.int_to_tag.items():
                    word_alt_features[tag_id] = self.feature_extractor(self.train_sentences[i], tag,
                                                                       self.train_tags[i][idx - 1] if idx > 0 else '*',
                                                                       self.train_tags[i][idx - 2] if idx > 1 else '*',
                                                                       idx)
                self.data_alt_features.append(sparse_mat(np.array(word_alt_features)))
        self.data_features = sparse_mat(self.data_features)

    def calculate_probability(self, tag, words, tags, idx, v, mahane):

        mone = exp(v.dot(self.feature_extractor(words, tag, tags[idx - 1] if idx > 0 else '*',
                                                             tags[idx - 2] if idx > 1 else '*', idx)))
        return mone / mahane

    def calculate_log_probability(self, tag, words, tags, idx, v, mahane):
        mone = sum(v.dot(self.feature_extractor(words, tag, tags[idx - 1] if idx > 0 else '*',
                                                             tags[idx - 2] if idx > 1 else '*', idx)))
        return mone - log(mahane)

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
        return -np.subtract(np.subtract(self.sum_of_features, exp_count.toarray()[:,0]), (LAMBDA * v_s).transpose().toarray()[:,0])

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
        logger.debug("Result of minimize is {}".format(d['warnflag']))
        logger.debug("Function called {} times".format(d['funcalls']))
        logger.debug("Number of iterations {}".format(d['nit']))
        np.save('v', self.v)


if __name__ == '__main__':
    all_sentences = prs.parse('train.wtag')
    mymodel = Model(all_sentences).train()
