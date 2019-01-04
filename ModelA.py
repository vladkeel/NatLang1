import data_parser as prs
from Model import Model, progress_bar
from scipy.sparse import csr_matrix as sparse_mat
import numpy as np


class ModelA(Model):

    def feature_collector(self, words, tags, current_tag, idx):
        key = 'includes_numeral_{}'.format(current_tag)
        if key not in self.set_of_features:
            self.set_of_features[key] = 1
        else:
            self.set_of_features[key] += 1
        key = 'capitalized_{}'.format(current_tag)
        if key not in self.set_of_features:
            self.set_of_features[key] = 1
        else:
            self.set_of_features[key] += 1
        if current_tag not in self.set_of_features:
            self.set_of_features[current_tag] = 1
        else:
            self.set_of_features[current_tag] += 1
        key = '{}_{}'.format(words[idx], current_tag)
        if key not in self.set_of_features:
            self.set_of_features[key] = 1
        else:
            self.set_of_features[key] += 1
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'suffix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                if key not in self.set_of_features:
                    self.set_of_features[key] = 1
                else:
                    self.set_of_features[key] += 1
        for i in range(1, 5):
            if len(words[idx]) >= i:
                key = 'prefix{}_{}_{}'.format(i, words[idx][-i:], current_tag)
                if key not in self.set_of_features:
                    self.set_of_features[key] = 1
                else:
                    self.set_of_features[key] += 1

        if idx == 0:
            key = '{}_{}_{}'.format('*', '*', current_tag)
        elif idx == 1:
            key = '{}_{}_{}'.format('*', tags[idx - 1], current_tag)
        else:
            key = '{}_{}_{}'.format(tags[idx - 2], tags[idx - 1], current_tag)

        if key not in self.set_of_features:
            self.set_of_features[key] = 1
        else:
            self.set_of_features[key] += 1

        if idx == 0:
            key = '{}_{}'.format('*', current_tag)
        else:
            key = '{}_{}'.format(tags[idx - 1], current_tag)

        if key not in self.set_of_features:
            self.set_of_features[key] = 1
        else:
            self.set_of_features[key] += 1

        if idx > 0:
            key = 'prev_{}_{}'.format(words[idx - 1], current_tag)
            if key not in self.set_of_features:
                self.set_of_features[key] = 1
            else:
                self.set_of_features[key] += 1

        if idx < len(words) - 1:
            key = 'next_{}_{}'.format(words[idx + 1], current_tag)
            if key not in self.set_of_features:
                self.set_of_features[key] = 1
            else:
                self.set_of_features[key] += 1

    def feature_extractor(self, words, is_cap, is_num, tag, last_t, last2_t, idx):
        return sparse_mat(self.feature_extractor_aux(words, is_cap, is_num, tag, last_t, last2_t, idx))

    def feature_extractor_all_tags(self, words, is_cap, is_num, last_t, last2_t, idx):
        word_alt_features = [None] * len(self.set_of_tags)
        for tag_id, tag in self.int_to_tag.items():
            word_alt_features[tag_id] = self.feature_extractor_aux(words, is_cap, is_num, tag, last_t, last2_t, idx)
        return sparse_mat(word_alt_features)

    def feature_extractor_all_tags2(self, words, is_cap, is_num, last_t, last2_t, idx):
        word_alt_features = [None] * len(self.set_of_tags)
        for tag_id, tag in self.int_to_tag.items():
            word_alt_features[tag_id] = self.feature_extractor_aux(words, is_cap, is_num, last_t, last2_t, tag, idx)
        return sparse_mat(word_alt_features)

    def feature_extractor_for_tags(self, words, is_cap, is_num, tags, last_t, last2_t, idx):
        mat = []
        for t in tags:
            mat.append(self.feature_extractor_aux(words, is_cap, is_num, last_t, last2_t, t, idx))
        return sparse_mat(mat).transpose()

    def feature_extractor_aux(self, words, is_cap, is_num, tag, last_t, last2_t, idx):
        new_ret = np.zeros(self.int)
        if is_cap[idx] and 'capitalized_{}'.format(tag) in self.key_to_int:
            new_ret[self.key_to_int['capitalized_{}'.format(tag)]] = 1
        if is_num[idx] and 'includes_numeral_{}'.format(tag) in self.key_to_int:
            new_ret[self.key_to_int['includes_numeral_{}'.format(tag)]] = 1
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

    def test(self, train_file, test_file,f):
        train_data = prs.parse(train_file)
        self.train(train_data)
        test = prs.parse(test_file)
        cnf_matrix = np.zeros((len(self.set_of_tags), len(self.set_of_tags)))
        for i, sentence in enumerate(test, start=1):
            tags = [a[1] for a in sentence]
            tags_result = self.infer(sentence)
            for j in range(len(tags)):
                if tags[j] in self.set_of_tags and tags_result[j] in self.set_of_tags:
                    cnf_matrix[self.tag_to_int[tags[j]]][self.tag_to_int[tags_result[j]]] += 1
            progress_bar(i / len(test), " Inferring sentence: {} from: {}".format(i, len(test)))
        print()
        sum_good = sum([cnf_matrix[i][i] for i in range(len(self.set_of_tags))])
        sum_all = cnf_matrix.sum()
        result_accuracy = sum_good/sum_all
        print("Confusion matrix:", file=f)
        str_mat = '\n'.join(' '.join('%0.0f' %x for x in y) for y in cnf_matrix)
        print(str_mat, file=f)
        print("Accuracy: {}".format(result_accuracy), file=f)
