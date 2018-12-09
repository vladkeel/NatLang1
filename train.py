from parser import parse
from math import exp, log
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
import datetime
import logging
import sys
import functools
import time

FORMAT = '%(asctime)-15s %(message)s'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formater = logging.Formatter(FORMAT)
handler.setFormatter(formater)
logger.addHandler(handler)
LAMBDA = 0
iteration_number = 1


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {str(datetime.timedelta(seconds=run_time))}")
        return value
    return wrapper_timer


class Model:
    def __init__(self, train_data):
        self.set_of_tags = set()
        self.train_sentences = []
        self.train_tags = []
        self.key_to_int = defaultdict(lambda: 0)
        self.int = 1
        for sentence in train_data:
            words = [a[0] for a in sentence]
            self.train_sentences.append(words)
            tags = [a[1] for a in sentence]
            self.train_tags.append(tags)
            for tag in tags:
                self.set_of_tags.add(tag)
            for idx, _ in enumerate(sentence):
                self.feature_collector(words, tags, idx)
        self.sum_of_features = np.zeros(self.int)
        for sentence in train_data:
            words = [a[0] for a in sentence]
            self.train_sentences.append(words)
            tags = [a[1] for a in sentence]
            self.train_tags.append(tags)
            for tag in tags:
                self.set_of_tags.add(tag)
            for idx, _ in enumerate(sentence):
                feature = self.feature_extractor(words, tags, idx)
                for i in feature:
                    self.sum_of_features[i] += 1

    def calculate_probability(self, tag, words, tags, idx, v, mahane):

        mone = exp(sum([v[x] for x in self.feature_extractor(words, tags, idx, tag)]))
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
        empirical_count = 0
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                empirical_count += sum([v[x] for x in self.feature_extractor(self.train_sentences[i], self.train_tags[i], j)])
        expected_counts = 0
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                sum_over_tags = 0
                for tag in self.set_of_tags:
                    sum_over_tags += exp(sum([v[x] for x in self.feature_extractor(self.train_sentences[i], self.train_tags[i], j, tag)]))
                expected_counts += log(sum_over_tags)
        logger.info('v is now: {}'.format(v))
        return -(empirical_count - expected_counts)# - ((LAMBDA/2) * np.dot(v, v)))

    @timer
    def dLdv(self, v):
        logger.info("run dldv")
        expected_counts = np.zeros(self.int)
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                mahane = 0
                for current_tag in self.set_of_tags:
                    mahane += exp(sum([v[x] for x in self.feature_extractor(
                        self.train_sentences[i], self.train_tags[i], j, current_tag)]))
                for tag in self.set_of_tags:
                    features = self.feature_extractor(self.train_sentences[i], self.train_tags[i], j, tag)
                    prob = self.calculate_probability(tag, self.train_sentences[i], self.train_tags[i], j, v, mahane)
                    for idx in features:
                        expected_counts[idx] += prob
        logger.info('v is now: {}'.format(v))
        return -(self.sum_of_features - expected_counts) #- (LAMBDA*v))

    def feature_extractor(self, words, tags, idx, current_tag=None):
        # print('run feature extractor')
        if current_tag is None:
            current_tag = tags[idx]
        new_ret = []
        key = '{}_{}'.format(words[idx], current_tag)
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'suffix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                new_ret.append(self.key_to_int[key])

        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'prefix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                new_ret.append(self.key_to_int[key])

        if idx == 0:
            key = '{}_{}_{}'.format('*', '*', current_tag)
        elif idx == 1:
            key = '{}_{}_{}'.format('*', tags[idx - 1], current_tag)
        else:
            key = '{}_{}_{}'.format(tags[idx - 2], tags[idx - 1], current_tag)

        new_ret.append(self.key_to_int[key])

        if idx == 0:
            key = '{}_{}'.format('*', current_tag)
        else:
            key = '{}_{}'.format(tags[idx - 1], current_tag)

        new_ret.append(self.key_to_int[key])

        new_ret.append(self.key_to_int[current_tag])

        if idx > 0:
            key = 'prev_{}_{}'.format(words[idx - 1], current_tag)
            new_ret.append(self.key_to_int[key])

        if idx < len(words) - 1:
            key = 'next_{}_{}'.format(words[idx + 1], current_tag)
            new_ret.append(self.key_to_int[key])

        return new_ret

    def train(self):
        print('start time: {}'.format(datetime.datetime.now()))
        self.v = minimize(self.L, np.zeros(self.int), method='L-BFGS-B', jac=self.dLdv)
        print('end time: {}'.format(datetime.datetime.now()))
        self.v.dump('v.dat')


if __name__ == '__main__':
    all_sentences = parse('train.wtag')
    mymodel = Model(all_sentences).train()
    a=1


