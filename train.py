from parser import parse
from math import exp, log
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

LAMBDA = 6


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
                self.sum_of_features = np.add(self.sum_of_features, self.feature_extractor(words, tags, idx))
        a=1

    def calculate_probability(self, tag, words, tags, idx, v):

        mone = exp(np.dot(v, self.feature_extractor(words, tags, idx, tag)))
        mahane = 0
        for current_tag in self.set_of_tags:
            mahane += exp(np.dot(v, self.feature_extractor(words, tags, idx, current_tag)))

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

    def L(self, v):
        empirical_count = 0
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                dotprod = np.dot(v, self.feature_extractor(self.train_sentences[i], self.train_tags[i], j))
                empirical_count += dotprod

        expected_counts = 0
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                sum_over_tags = 0
                for tag in self.set_of_tags:
                    sum_over_tags += exp(np.dot(
                        v, self.feature_extractor(self.train_sentences[i], self.train_tags[i], j, tag)))
                expected_counts += log(sum_over_tags)

        return empirical_count - expected_counts - ((LAMBDA/2) * np.dot(v, v))

    def dLdv(self, v):

        expected_counts = np.zeros(self.int)
        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                for tag in self.set_of_tags:
                    features = self.feature_extractor(self.train_sentences[i], self.train_tags[i], j, tag)
                    prob = self.calculate_probability(tag, self.train_sentences[i], self.train_tags[i], j, v)
                    features = features * prob
                    expected_counts = np.add(expected_counts, features)

        return self.sum_of_features - expected_counts - (LAMBDA*v)

    def feature_extractor(self, words, tags, idx, current_tag=None):

        if current_tag is None:
            current_tag = tags[idx]

        ret = np.zeros(self.int)
        key = '{}_{}'.format(words[idx], current_tag)
        ret[self.key_to_int[key]] = 1
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'suffix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                ret[self.key_to_int[key]] = 1

        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'prefix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                ret[self.key_to_int[key]] = 1

        if idx == 0:
            key = '{}_{}_{}'.format('*', '*', current_tag)
        elif idx == 1:
            key = '{}_{}_{}'.format('*', tags[idx - 1], current_tag)
        else:
            key = '{}_{}_{}'.format(tags[idx - 2], tags[idx - 1], current_tag)

        ret[self.key_to_int[key]] = 1

        if idx == 0:
            key = '{}_{}'.format('*', current_tag)
        else:
            key = '{}_{}'.format(tags[idx - 1], current_tag)

        ret[self.key_to_int[key]] = 1

        ret[self.key_to_int[current_tag]] = 1

        if idx > 0:
            key = 'prev_{}_{}'.format(words[idx - 1], current_tag)
            ret[self.key_to_int[key]] = 1

        if idx < len(words) - 1:
            key = 'next_{}_{}'.format(words[idx + 1], current_tag)
            ret[self.key_to_int[key]] = 1

        return ret

    def train(self):
        self.v = minimize(self.L, np.zeros(self.int), method='L-BFGS-B', jac=self.dLdv)
        self.v.dump('v.dat')






all_sentences = parse('train.wtag')
mymodel = Model(all_sentences).train()
a=1


