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
import pickle
import operator
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
    def __init__(self, train_data, is_test=False):
        if is_test:
            self.v = np.load('v.npy')
            with open('int_to_tag', 'rb') as f:
                self.int_to_tag = pickle.load(f)
            with open('set_of_tags', 'rb') as f:
                self.set_of_tags = pickle.load(f)
            with open('tag_to_int', 'rb') as f:
                self.tag_to_int = pickle.load(f)
            with open('key_to_int', 'rb') as f:
                self.key_to_int = pickle.load(f)
            with open('word_tag_dict', 'rb') as f:
                self.word_tag_dict = pickle.load(f)
            self.int = len(self.key_to_int)
            return
        logger.info("Initializing model")
        self.word_tag_dict = {}
        self.iteration_number = 1
        self.set_of_tags = set()
        self.train_sentences = []
        self.train_tags = []
        self.train_is_cap = []
        self.train_is_num = []
        self.key_to_int = {}
        self.int = 0
        tag_enum = 0
        self.tag_to_int = {}
        self.int_to_tag = {}
        self.pi = None
        self.bp = None
        logger.info("Collecting features and tags")
        for sentence in train_data:
            words = [a[0] for a in sentence]
            tags = [a[1] for a in sentence]
            is_cap = [a[2] for a in sentence]
            is_num = [a[3] for a in sentence]
            self.train_sentences.append(words)
            self.train_tags.append(tags)
            self.train_is_cap.append(is_cap)
            self.train_is_num.append(is_num)
            for tag in tags:
                if tag not in self.set_of_tags:
                    self.set_of_tags.add(tag)
                    self.tag_to_int[tag] = tag_enum
                    self.int_to_tag[tag_enum] = tag
                    tag_enum += 1
            for idx, _ in enumerate(sentence):
                if words[idx] not in self.word_tag_dict:
                    self.word_tag_dict[words[idx]] = set([tags[idx]])
                else:
                    self.word_tag_dict[words[idx]].add(tags[idx])
                self.feature_collector(words, tags, idx)
        if '*' not in self.set_of_tags:
            self.set_of_tags.add('*')
            self.tag_to_int['*'] = tag_enum
            self.int_to_tag[tag_enum] = '*'
            tag_enum += 1

        with open('set_of_tags', 'wb') as f:
            pickle.dump(self.set_of_tags, f)
        with open('tag_to_int', 'wb') as f:
            pickle.dump(self.tag_to_int, f)
        with open('int_to_tag', 'wb') as f:
            pickle.dump(self.int_to_tag, f)
        with open('key_to_int', 'wb') as f:
            pickle.dump(self.key_to_int, f)
        with open('word_tag_dict', 'wb') as f:
            pickle.dump(self.word_tag_dict, f)
        logger.info("Collected {} features and {} tags".format(self.int, tag_enum))
        self.data_features = sparse_mat((0, self.int))
        self.data_alt_features = []
        self.v = None
        self.sum_of_features = sparse_mat((1, self.int))
        logger.info("Extracting features")
        for i in range(len(self.train_sentences)):
            for idx in range(len(self.train_sentences[i])):
                fv = sparse_mat(self.feature_extractor(
                    self.train_sentences[i], self.train_is_cap[i], self.train_is_num[i], self.train_tags[i][idx],
                    self.train_tags[i][idx - 1] if idx > 0 else '*',
                    self.train_tags[i][idx - 2] if idx > 1 else '*', idx))
                self.data_features = hstack([self.data_features.transpose(), fv.transpose()], format='csr').transpose()
                self.sum_of_features += fv
                word_alt_features = [None] * len(self.set_of_tags)
                for tag_id, tag in self.int_to_tag.items():
                    word_alt_features[tag_id] = self.feature_extractor(self.train_sentences[i], self.train_is_cap[i],
                                                                       self.train_is_num[i], tag,
                                                                       self.train_tags[i][idx - 1] if idx > 0 else '*',
                                                                       self.train_tags[i][idx - 2] if idx > 1 else '*',
                                                                       idx)
                self.data_alt_features.append(sparse_mat(np.array(word_alt_features)))
            progress_bar(i/len(self.train_sentences), "completed {} of {} sentences".format(i, len(self.train_sentences)))
        logger.info("Extracted features for all words")

    def feature_collector(self, words, tags, idx):
        key = 'includes_numeral'
        if key not in self.key_to_int:
            self.key_to_int[key] = self.int
            self.int += 1
        key = 'capitalized'
        if key not in self.key_to_int:
            self.key_to_int[key] = self.int
            self.int += 1
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

    def feature_extractor(self, words, is_cap, is_num, tag, last_t, last2_t, idx):
        new_ret = np.zeros(self.int)
        if is_cap[idx]:
            new_ret[self.key_to_int['capitalized']] = 1
        if is_num[idx]:
            new_ret[self.key_to_int['includes_numeral']] = 1
        key = '{}_{}'.format(words[idx], tag)
        if key in self.key_to_int:
            new_ret[self.key_to_int[key]] = 1
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

    def tags_for_word(self, words, idx):
        if idx <= 0:
            return set('*')
        else:
            if words[idx-1] in self.word_tag_dict:
                return self.word_tag_dict[words[idx-1]]
            else:
                return self.set_of_tags

    def infer(self, words):
        logger.debug('Infering for given sentence')
        self.pi = np.zeros((len(words) + 1, len(self.set_of_tags), len(self.set_of_tags)))
        self.bp = np.zeros((len(words) + 1, len(self.set_of_tags), len(self.set_of_tags)))
        self.pi[0, self.tag_to_int['*'], self.tag_to_int['*']] = 1
        sentence = [words[i][0] for i in range(len(words))]
        is_cap = [words[i][2] for i in range(len(words))]
        is_num = [words[i][3] for i in range(len(words))]

        for k in range(1, len(words)+1):
            for u in self.tags_for_word(words, k-1):
                for v in self.tags_for_word(words, k):
                    logger.debug("building matrix for word {}, tag {}, last tag {}".format(k, v, u))
                    feature_tag_mat = sparse_mat((0, self.int))
                    for i in range(len(self.set_of_tags)):
                        feature_tag_mat = hstack([feature_tag_mat.transpose(), sparse_mat(
                            self.feature_extractor(sentence, is_cap, is_num, v, u, self.int_to_tag[i], k-1)).transpose()],
                               format='csr').transpose()
                    mahane = sum(np.exp(feature_tag_mat.dot(sparse_mat(self.v).transpose()).toarray()))
                    mone = np.exp(sparse_mat(self.v).dot(feature_tag_mat.transpose()).toarray())[0, :]
                    prob = mone / mahane
                    calc = [self.pi[k-1, idx, self.tag_to_int[u]] * prob[idx] for idx in range(len(self.set_of_tags))]
                    self.pi[k, self.tag_to_int[u], self.tag_to_int[v]] = max(calc)
                    self.bp[k, self.tag_to_int[u], self.tag_to_int[v]] = np.argmax(calc)

        tags = [None] * len(words)
        tags[len(tags)-1] = self.int_to_tag[np.argmax(self.pi[len(tags)])[1]]
        tags[len(tags)-2] = self.int_to_tag[np.argmax(self.pi[len(tags)])[0]]
        for k in range(len(words) - 3, -1, -1):
            tags[k] = self.int_to_tag[self.bp[k+3, self.tag_to_int[tags[k+1]], self.tag_to_int[tags[k+2]]]]
        return tags



if __name__ == '__main__':

    #all_sentences = prs.parse('train.wtag')
    #mymodel = Model(all_sentences, True)
    # mymodel.train()
    test = prs.parse('test.wtag')
    mymodel = Model([], True)
    with open('result_stats', 'w') as res_file:
        num_of_words = 0
        sum_good = 0
        for sentence in test:
            num_of_words += len(sentence)
            words = [a[0] for a in sentence]
            tags = [a[1] for a in sentence]
            tags_result = mymodel.infer(sentence)
            res = [1 if tags[i] == tags_result[i] else 0 for i in range(len(words))]
            sum_good += sum(res)
            res_file.write("sentence: {}".format(words))
            res_file.write("    real tags: {}".format(tags))
            res_file.write("    infr tags: {}".format(tags_result))
            res_file.write("    Percision: {}".format(sum(res) / len(words)))
        res_file.write("----------------------------------------------------------------------------------------------")
        res_file.write("Overall percision: {}".format(sum_good / num_of_words))
