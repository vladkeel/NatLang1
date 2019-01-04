import data_parser as prs
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
import os
logging.basicConfig(filename='logger.txt', level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)
LAMBDA = 10
iteration_number = 1
known_tags = {',':',', '.':'.', ':':':', "''":"''", ';':':', '``':'``', '#':'#', '--':':', '...':':', '-':':', "'":"''", '`':'``',
        '-rcb-':'-RRB-', '-rrb-':'-RRB-', '-lcb-':'-LRB-', '-lrb-':'-LRB-', '?':'.', '!':'.'}
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
    def __init__(self, dir, is_test=False):
        self.dir = dir
        self.is_test = is_test
        if is_test:
            self.v = np.load(os.path.join(self.dir, 'v.npy'))
            with open(os.path.join(dir, 'int_to_tag'), 'rb') as f:
                self.int_to_tag = pickle.load(f)
            with open(os.path.join(dir, 'set_of_tags'), 'rb') as f:
                self.set_of_tags = pickle.load(f)
            with open(os.path.join(dir, 'tag_to_int'), 'rb') as f:
                self.tag_to_int = pickle.load(f)
            with open(os.path.join(dir, 'key_to_int'), 'rb') as f:
                self.key_to_int = pickle.load(f)
            with open(os.path.join(dir, 'word_tag_dict'), 'rb') as f:
                self.word_tag_dict = pickle.load(f)
            self.int = len(self.key_to_int)
            return
        logger.info("Initializing model")
        self.word_tag_dict = {}
        self.iteration_number = 1
        self.set_of_tags = set()
        self.set_of_features = {}
        self.train_sentences = []
        self.train_tags = []
        self.train_is_cap = []
        self.train_is_num = []
        self.key_to_int = {}
        self.data_alt_features = []
        self.data_features = None
        self.sum_of_features = None
        self.int = 0
        self.tag_to_int = {}
        self.int_to_tag = {}
        self.pi = None
        self.bp = None
        self.v = None

    def clear(self):
        self.is_test = False
        self.word_tag_dict = {}
        self.iteration_number = 1
        self.set_of_tags = set()
        self.set_of_features = {}
        self.train_sentences = []
        self.train_tags = []
        self.train_is_cap = []
        self.train_is_num = []
        self.key_to_int = {}
        self.data_alt_features = []
        self.data_features = None
        self.sum_of_features = None
        self.int = 0
        self.tag_to_int = {}
        self.int_to_tag = {}
        self.pi = None
        self.bp = None
        self.v = None

    def feature_collector(self, words, tags, current_tag, idx):
        pass

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
        pass

    def feature_extractor_all_tags(self, words, is_cap, is_num, last_t, last2_t, idx):
        pass

    def feature_extractor_all_tags2(self, words, is_cap, is_num, last_t, last2_t, idx):
        pass

    def feature_extractor_for_tags(self, words, is_cap, is_num, tags, last_t, last2_t, idx):
        pass

    def feature_extractor_aux(self, words, is_cap, is_num, tag, last_t, last2_t, idx):
        pass

    def finish_train(self):
        tag_idx = len(self.set_of_tags)
        self.int_to_tag[tag_idx] = '*'
        self.tag_to_int['*'] = tag_idx
        with open(os.path.join(self.dir, 'set_of_tags'), 'wb') as f:
            pickle.dump(self.set_of_tags, f)
        with open(os.path.join(self.dir, 'tag_to_int'), 'wb') as f:
            pickle.dump(self.tag_to_int, f)
        with open(os.path.join(self.dir, 'int_to_tag'), 'wb') as f:
            pickle.dump(self.int_to_tag, f)
        with open(os.path.join(self.dir, 'key_to_int'), 'wb') as f:
            pickle.dump(self.key_to_int, f)
        with open(os.path.join(self.dir, 'word_tag_dict'), 'wb') as f:
            pickle.dump(self.word_tag_dict, f)
        np.save(os.path.join(self.dir, 'v'), self.v)

    def train(self, train_data):
        if self.is_test:
            logger.info("Model already trained.")
            return
        logger.info("Collecting features and tags")
        tag_enum = 0
        for sentence in train_data:
            filter_sentence = [x for x in sentence if x[0] not in known_tags]
            words = [a[0] for a in filter_sentence]
            tags = [a[1] for a in filter_sentence]
            is_cap = [a[2] for a in filter_sentence]
            is_num = [a[3] for a in filter_sentence]
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
            for idx, _ in enumerate(filter_sentence):
                if words[idx] not in self.word_tag_dict:
                    self.word_tag_dict[words[idx]] = {tags[idx]}
                else:
                    self.word_tag_dict[words[idx]].add(tags[idx])
        for i in range(len(self.train_sentences)):
            for idx in range(len(self.train_sentences[i])):
                self.feature_collector(self.train_sentences[i], self.train_tags[i], self.train_tags[i][idx], idx)
            progress_bar(i / len(self.train_sentences),
                         "completed {} of {} sentences".format(i, len(self.train_sentences)))
        set_of_useful_features = [k for k, v in self.set_of_features.items() if float(v) >= 5]
        self.int = 0
        for key in set_of_useful_features:
            self.key_to_int[key] = self.int
            self.int += 1

        logger.info("Collected {} features and {} tags".format(self.int, tag_enum))
        self.data_features = sparse_mat((0, self.int))
        self.data_alt_features = []
        self.sum_of_features = sparse_mat((1, self.int))
        logger.info("Extracting features")
        for i in range(len(self.train_sentences)):
            for idx in range(len(self.train_sentences[i])):
                fv = self.feature_extractor(self.train_sentences[i], self.train_is_cap[i], self.train_is_num[i],
                                            self.train_tags[i][idx],
                                            self.train_tags[i][idx - 1] if idx > 0 else '*',
                                            self.train_tags[i][idx - 2] if idx > 1 else '*', idx)
                self.data_features = hstack([self.data_features.transpose(), fv.transpose()], format='csr').transpose()
                self.sum_of_features += fv
                self.data_alt_features.append(self.feature_extractor_all_tags(self.train_sentences[i],
                                                                              self.train_is_cap[i],
                                                                              self.train_is_num[i],
                                                                              self.train_tags[i][idx - 1] if idx > 0 else '*',
                                                                              self.train_tags[i][idx - 2] if idx > 1 else '*',
                                                                              idx))
            progress_bar(i/len(self.train_sentences), "completed {} of {} sentences".format(i, len(self.train_sentences)))
        logger.info("Extracted features for all words")
        logger.debug('Start Now!!')
        self.v, f, d = minimize(self.L, np.zeros(self.int), factr=1e12, pgtol=1e-3, fprime=self.dLdv)
        logger.debug('End Now!!')
        logger.debug("v is: {}".format(self.v))
        logger.debug("Result of minimize is {}".format("success" if d['warnflag'] == 0 else "failure"))
        logger.debug("Function called {} times".format(d['funcalls']))
        logger.debug("Number of iterations {}".format(d['nit']))
        self.finish_train()

    def tags_for_word(self, words, idx):
        if idx <= 0:
            return set('*')
        else:
            if words[idx-1] in self.word_tag_dict:
                return self.word_tag_dict[words[idx-1]]
            else:
                return self.set_of_tags

    def infer(self, all_words):
        # logger.debug('Infering for given sentence')
        filter_words = [x for x in all_words if x[0] not in known_tags]
        ret_tags = [None]*len(all_words)
        for i in range(len(all_words)):
            if all_words[i][0] in known_tags:
                ret_tags[i] = known_tags[all_words[i][0]]
        self.pi = np.zeros((len(filter_words) + 1, len(self.tag_to_int), len(self.tag_to_int)))
        self.bp = np.zeros((len(filter_words) + 1, len(self.tag_to_int), len(self.tag_to_int)))
        self.pi[0, self.tag_to_int['*'], self.tag_to_int['*']] = 1
        sentence = [filter_words[i][0] for i in range(len(filter_words))]
        is_cap = [filter_words[i][2] for i in range(len(filter_words))]
        is_num = [filter_words[i][3] for i in range(len(filter_words))]
        v_s = sparse_mat(self.v)
        for k in range(1, len(sentence)+1):
            V = {x: i for i, x in enumerate(self.tags_for_word(sentence, k))}
            U = {x: i for i, x in enumerate(self.tags_for_word(sentence, k-1))}
            T = {x: i for i, x in enumerate(self.tags_for_word(sentence, k-2))}
            for u in U.keys():
                feature_tag_mat_per_v = [self.feature_extractor_for_tags(sentence, is_cap, is_num,
                                                                         T.keys(), v, u, k - 1) for v in V.keys()]
                mahane = [np.sum(np.exp(v_s.dot(sparse_mat([feature_tag_mat_per_v[v][:,
                                                                T[t]].transpose().toarray()[0, :] for v in V.values()])
                                                    .transpose()).toarray()[0, :])) for t in T.keys()]
                for v in V.keys():
                    mone = np.exp(v_s.dot((feature_tag_mat_per_v[V[v]])).toarray())[0, :]
                    prob = mone / mahane
                    calc = [self.pi[k-1, self.tag_to_int[t], self.tag_to_int[u]] * prob[idx] for t, idx in T.items()]
                    self.pi[k, self.tag_to_int[u], self.tag_to_int[v]] = max(calc)
                    tmp = list(T.keys())[list(T.values()).index(np.argmax(calc))]
                    self.bp[k, self.tag_to_int[u], self.tag_to_int[v]] = self.tag_to_int[tmp]

        tags = [None] * len(sentence)
        u, v = np.unravel_index(np.argmax(self.pi[len(sentence)]), self.pi[len(sentence)].shape)
        tags[len(tags)-1] = self.int_to_tag[v]
        tags[len(tags)-2] = self.int_to_tag[u]
        for k in range(len(sentence) - 3, -1, -1):
            tags[k] = self.int_to_tag[self.bp[k+3, self.tag_to_int[tags[k+1]], self.tag_to_int[tags[k+2]]]]
        i = 0
        for tag in tags:
            while ret_tags[i] is not None:
                i += 1
            ret_tags[i] = tag
        return ret_tags

    def test(self, train_file, test_file, f):
        pass

    def comp(self, comp_file):
        test = prs.comp_pars(comp_file)
        res_rows = []
        iterat = 1
        for sentence in test:
            words = [sentence[i][0] for i in range(len(sentence))]
            tags = self.infer(sentence)
            res_rows.append(prs.results_row(words, tags))
            progress_bar(iterat/len(test), "Completed {} of {} competition sentences".format(iterat, len(test)))
            iterat += 1
        prs.write_results('res_{}'.format(comp_file), res_rows)

