from parser import parse
from math import exp, log
import numpy as np
from collections import defaultdict
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
        for sentence in train_data:
            words = [a[0] for a in sentence]
            self.train_sentences.append(words)
            tags = [a[1] for a in sentence]
            self.train_tags.append(tags)
            for tag in tags:
                self.set_of_tags.add(tag)
            for idx, _ in enumerate(sentence):
                self.feature_collector(words, tags, idx)
        self.sum_of_features = sparse_mat(1, self.int)
        self.v = sparse_mat(1, self.int)
        for sentence in train_data:
            words = [a[0] for a in sentence]
            self.train_sentences.append(words)
            tags = [a[1] for a in sentence]
            self.train_tags.append(tags)
            for tag in tags:
                self.set_of_tags.add(tag)
            for idx, _ in enumerate(sentence):
                feature = self.feature_extractor(words, tags[idx], tags[idx - 1] if idx > 0 else '*',
                                                 tags[idx - 2] if idx > 1 else '*', idx)
                self.sum_of_features = self.sum_of_features + feature
        enumerator = 0
        for tag in self.set_of_tags:
            self.tag_to_idx[tag] = enumerator
            enumerator += 1
        self.tag_to_idx['*'] = enumerator
        self.tag_to_idx['STOP'] = enumerator + 1

    def calculate_probability(self, tag, words, tags, idx, v, mahane):

        mone = exp(v.dot(self.feature_extractor(words, tag, tags[idx - 1] if idx > 0 else '*',
                                                             tags[idx - 2] if idx > 1 else '*', idx)))
        return mone / mahane

    def calculate_log_probability(self, tag, words, tags, idx, v, mahane):
        try:
            mone = sum(v.dot(self.feature_extractor(words, tag, tags[idx - 1] if idx > 0 else '*',
                                                             tags[idx - 2] if idx > 1 else '*', idx)))
            return mone - log(mahane)
        except:
            logger.critical("mahane is negative: {}".format(mahane))

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
        empirical_count = 0
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                mahane = 0
                for current_tag in self.set_of_tags:
                    mahane += exp(v.dot(self.feature_extractor(
                        self.train_sentences[i], current_tag, self.train_tags[i][j-1] if j > 0 else '*',
                        self.train_tags[i][j-2] if j > 1 else '*', j)))
                empirical_count += self.calculate_log_probability(self.train_tags[i][j], self.train_sentences[i],
                                                                  self.train_tags[i], j, v, mahane)
        # logger.info('v is now: {}'.format(v))
        return -(empirical_count - ((LAMBDA / 2) * v.dot(v)))

    @timer
    def dLdv(self, v):
        logger.info("run dldv")
        expected_counts = sparse_mat(1, self.int)
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                mahane = 0
                for current_tag in self.set_of_tags:
                    mahane += exp(v.dot(self.feature_extractor(
                        self.train_sentences[i], current_tag, self.train_tags[i][j-1] if j > 0 else '*',
                        self.train_tags[i][j-2] if j > 1 else '*', j)))
                for tag in self.set_of_tags:
                    features = self.feature_extractor(
                        self.train_sentences[i], tag, self.train_tags[i][j-1] if j > 0 else '*',
                        self.train_tags[i][j-2] if j > 1 else '*', j)
                    prob = self.calculate_probability(tag, self.train_sentences[i], self.train_tags[i], j, v, mahane)
                    expected_counts = expected_counts + (features*prob)

        # logger.info('v is now: {}'.format(v))
        logger.critical("iteration number: {}".format(self.iteration_number))
        self.iteration_number += 1
        return -(self.sum_of_features - expected_counts - (LAMBDA * v))

    def feature_extractor(self, words, tag, last_t, last2_t, idx):
        # print('run feature extractor')
        new_ret = sparse_mat(1, self.int)
        key = '{}_{}'.format(words[idx], tag)
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'suffix{}_{}_{}'.format(i, words[idx][-i:], tag)
                if key in self.key_to_int:
                    new_ret[0][self.key_to_int[key]] = 1

        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'prefix{}_{}_{}'.format(i, words[idx][-i:], tag)
                if key in self.key_to_int:
                    new_ret[0][self.key_to_int[key]] = 1

        if idx == 0:
            key = '{}_{}_{}'.format('*', '*', tag)
        elif idx == 1:
            key = '{}_{}_{}'.format('*', last_t, tag)
        else:
            key = '{}_{}_{}'.format(last2_t, last_t, tag)

        if key in self.key_to_int:
            new_ret[0][self.key_to_int[key]] = 1

        if idx == 0:
            key = '{}_{}'.format('*', tag)
        else:
            key = '{}_{}'.format(last_t, tag)

        if key in self.key_to_int:
            new_ret[0][self.key_to_int[key]] = 1
        if tag in self.key_to_int:
            new_ret[0][self.key_to_int[tag]] = 1

        if idx > 0:
            key = 'prev_{}_{}'.format(words[idx - 1], tag)
            if key in self.key_to_int:
                new_ret[0][self.key_to_int[key]] = 1

        if idx < len(words) - 1:
            key = 'next_{}_{}'.format(words[idx + 1], tag)
            if key in self.key_to_int:
                new_ret[0][self.key_to_int[key]] = 1
        return new_ret

    def train(self):
        logger.debug('Start Now!!')
        self.v = minimize(self.L, sparse_mat(1, self.int), factr=1e12, pgtol=1e-3, fprime=self.dLdv)[0]
        logger.debug('End Now!!')
        logger.debug("v is: {}".format(self.v))
        np.save('v', self.v[0])
        np.savetxt('v.txt', self.v[0])


if __name__ == '__main__':
    all_sentences = parse('train.wtag')
    mymodel = Model(all_sentences).train()
