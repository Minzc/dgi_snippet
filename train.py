# -*- coding: utf-8 -*-
import argparse
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
import torch.optim

from data import MovieLens


def config():
    parser = argparse.ArgumentParser(description='GCMC')
    parser.add_argument('--device', default=0, type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--do_squeeze', action='store_true')

    args = parser.parse_args()

    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')

    return args


class GCMCGraphConv(nn.Module):

    def __init__(self,
                 node_count,
                 feat_size,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        self.weight = nn.Parameter(th.Tensor(node_count, feat_size))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, u_feat, i_feat):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)

            feat = self.weight

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCMCLayer(nn.Module):
    def __init__(self,
                 rating_vals,
                 user_count,
                 item_count,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.ufc = nn.Linear(msg_units, out_units)
        self.ifc = nn.Linear(msg_units, out_units)

        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        subConv = {}

        for rating in rating_vals:
            # PyTorch parameter name can't contain "."
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            self.W_r = None
            subConv[rating] = GCMCGraphConv(user_count,
                                            msg_units,
                                            device=device,
                                            dropout_rate=dropout_rate)
            subConv[rev_rating] = GCMCGraphConv(item_count,
                                                msg_units,
                                                device=device,
                                                dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate='sum')
        self.agg_act = nn.LeakyReLU(0.1)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, feat=None):
        """Forward function

        Parameters
        ----------
        graph : DGLHeteroGraph
            User-movie rating graph. It should contain two node types: "user"
            and "movie" and many edge types each for one rating value.

        Returns
        -------
        new_ufeat : torch.Tensor
            New user features
        new_ifeat : torch.Tensor
            New movie features
        """
        in_feats = {'user': None, 'movie': None}
        mod_args = {}
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            mod_args[rating] = (self.W_r[rating] if self.W_r is not None else None,)
            mod_args[rev_rating] = (self.W_r[rev_rating] if self.W_r is not None else None,)
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        ufeat = out_feats['user']
        ifeat = out_feats['movie']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        # return self.out_act(ufeat), self.out_act(ifeat)
        return ufeat, ifeat


class MLPPredictor(nn.Module):

    def __init__(self,
                 in_units,
                 do_squeeze=True,
                 dropout_rate=0.0):
        super(MLPPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.predictor = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.Tanh(),
            nn.Linear(in_units, 1, bias=False),
        )
        self.do_squeeze = do_squeeze

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        if self.do_squeeze:
            score = self.predictor(th.cat([h_u, h_v], dim=1)).squeeze()
        else:
            score = self.predictor(th.cat([h_u, h_v], dim=1))

        return {'score': score}

    def forward(self, graph, ufeat, ifeat):
        graph.nodes['movie'].data['h'] = ifeat
        graph.nodes['user'].data['h'] = ufeat
        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.encoder = GCMCLayer(args.rating_vals,
                                 args.user_count,
                                 args.item_count,
                                 500,
                                 75,
                                 dropout_rate=0.8,
                                 device=args.device)
        self.decoder = MLPPredictor(in_units=75,
                                    do_squeeze=args.do_squeeze)

    def forward(self, enc_graph, dec_graph):
        user_out, movie_out = self.encoder(enc_graph)
        pred_ratings = self.decoder(dec_graph, user_out, movie_out)
        return pred_ratings


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def train(params):
    dataset = MovieLens("ml-100k", device=params.device, symm=True)
    print("Loading data finished ...\n")

    params.user_count = dataset.num_user
    params.item_count = dataset.num_movie
    params.rating_vals = dataset.possible_rating_values

    # build the net
    net = Net(args=params)
    net = net.to(params.device)
    # nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(params.device)

    rating_loss_net = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    print("Loading network finished ...\n")

    # perpare training data
    train_gt_labels = dataset.train_labels.to(torch.float32)

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(params.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(params.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)

    for iter_idx in range(1, 2000):
        net.train()
        pred_ratings = net(dataset.train_enc_graph, dataset.train_dec_graph)
        loss = rating_loss_net(pred_ratings, train_gt_labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        print(loss.item())


if __name__ == '__main__':
    train(config())

