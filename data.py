# -*- coding: utf-8 -*-
"""MovieLens dataset"""
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir


ml100k_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'


def to_etype_name(rating):
    return str(rating).replace('.', '_')


class MovieLens(object):
    """MovieLens dataset used by GCMC model

    The dataset stores MovieLens ratings in two types of graphs. The encoder graph
    contains rating value information in the form of edge types. The decoder graph
    stores plain user-movie pairs in the form of a bipartite graph with no rating
    information. All graphs have two types of nodes: "user" and "movie".

    The training, validation and test set can be summarized as follows:

    training_enc_graph : training user-movie pairs + rating info
    training_dec_graph : training user-movie pairs
    valid_enc_graph : training user-movie pairs + rating info
    valid_dec_graph : validation user-movie pairs
    test_enc_graph : training user-movie pairs + validation user-movie pairs + rating info
    test_dec_graph : test user-movie pairs

    Attributes
    ----------
    train_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for training.
    train_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for training.
    train_labels : torch.Tensor
        The categorical label of each user-movie pair
    train_truths : torch.Tensor
        The actual rating values of each user-movie pair
    valid_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for validation.
    valid_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for validation.
    valid_labels : torch.Tensor
        The categorical label of each user-movie pair
    valid_truths : torch.Tensor
        The actual rating values of each user-movie pair
    test_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for test.
    test_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for test.
    test_labels : torch.Tensor
        The categorical label of each user-movie pair
    test_truths : torch.Tensor
        The actual rating values of each user-movie pair
    user_feature : torch.Tensor
        User feature tensor. If None, representing an identity matrix.
    movie_feature : torch.Tensor
        Movie feature tensor. If None, representing an identity matrix.
    possible_rating_values : np.ndarray
        Available rating values in the dataset

    Parameters
    ----------
    dataset_path : str
        Dataset name. Could be "ml-100k", "ml-1m", "ml-10m"
    device : torch.device
        Device context
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)

    """

    def __init__(self, name, device, symm=True,):
        self._device = device
        self._symm = symm

        download_dir = get_download_dir()
        zip_file_path = '{}/{}.zip'.format(download_dir, name)
        download(ml100k_url, path=zip_file_path)
        extract_archive(zip_file_path, '{}/{}'.format(download_dir, name))
        root_folder = name
        self._dir = os.path.join(download_dir, name, root_folder)

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = \
            self.load_ml100k(self._dir)

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            item_id = info['item_id'].to_list()
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self._num_user = dataset_info['user_size']
        self._num_movie = dataset_info['item_size']

        self.user_feature = None
        self.movie_feature = None

        print(f'#user: {self.num_user}, #movie: {self.num_movie}')

        train_rating_pairs, train_rating_values = self._generate_pair_value('train')
        valid_rating_pairs, valid_rating_values = self._generate_pair_value('valid')
        test_rating_pairs, test_rating_values = self._generate_pair_value('test')

        def _make_labels(ratings):
            """
            不同rating值对应id
            """
            labels = th.LongTensor(
                np.searchsorted(self.possible_rating_values, ratings)).to(
                device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs,
                                                        train_rating_values,
                                                        add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = _make_labels(valid_rating_values)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self.train_enc_graph
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

    def _generate_pair_value(self, sub_dataset):
        """
        :param sub_dataset: all, train, valid, test
        :return:
        """
        if sub_dataset == 'all_train':
            user_id = self.train_datas[0] + self.valid_datas[0]
            item_id = self.train_datas[1] + self.valid_datas[1]
            rating = self.train_datas[2] + self.valid_datas[2]
        elif sub_dataset == 'train':
            user_id = self.train_datas[0]
            item_id = self.train_datas[1]
            rating = self.train_datas[2]
        elif sub_dataset == 'valid':
            user_id = self.valid_datas[0]
            item_id = self.valid_datas[1]
            rating = self.valid_datas[2]
        else:
            user_id = self.test_datas[0]
            item_id = self.test_datas[1]
            rating = self.test_datas[2]

        rating_pairs = (np.array(user_id, dtype=np.int64),
                        np.array(item_id, dtype=np.int64))
        rating_values = np.array(rating, dtype=np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values,
                            add_support=False):
        user_movie_r = np.zeros((self._num_user, self._num_movie),
                                dtype=np.float32)
        user_movie_r[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum(
            [graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(self.num_user, )
                movie_cj = th.ones(self.num_movie, )
            graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
            graph.nodes['movie'].data.update({'ci': movie_ci, 'cj': movie_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U',
                                     etype='_E', vtype='_V')
        g = dgl.heterograph({('user', 'rate', 'movie'): g.edges()},
                            num_nodes_dict={'user': self.num_user,
                                            'movie': self.num_movie})

        return g

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    @staticmethod
    def load_ml100k(dataset_path):
        train_path = f'{dataset_path}/u1.base'
        test_path = f'{dataset_path}/u1.test'
        valid_ratio = 0.1
        train = pd.read_csv(
            train_path, sep='\t', header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int64, 'item_id': np.int64,
                   'ratings': np.float32, 'timestamp': np.int64},
            engine='python')
        user_size = train['user_id'].max() + 1
        item_size = train['item_id'].max() + 1
        dataset_info = {'user_size': user_size, 'item_size': item_size}
        test = pd.read_csv(
            test_path, sep='\t', header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int64, 'item_id': np.int64,
                   'ratings': np.float32, 'timestamp': np.int64},
            engine='python')
        num_valid = int(
            np.ceil(train.shape[0] * valid_ratio))
        shuffled_idx = np.random.permutation(train.shape[0])
        valid = train.iloc[shuffled_idx[: num_valid]]
        train = train.iloc[shuffled_idx[num_valid:]]

        return train, valid, test, None, None, dataset_info


if __name__ == '__main__':
    MovieLens("ml-100k", device=th.device('cpu'), symm=True)
