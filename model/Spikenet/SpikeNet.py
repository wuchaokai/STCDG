import torch
import torch.nn as nn
from model.Spikenet import dataset, neuron
from model.Spikenet.layers import SAGEAggregator
from model.Spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)
class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',adj=None,adj_evolve=None):

        super().__init__()

        tau = 1.0
        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(
                add_selfloops(adj_matrix)) for adj_matrix in adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(
                adj_matrix)) for adj_matrix in adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix))
                            for adj_matrix in adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix))
                              for adj_matrix in adj_evolve]
        else:
            raise ValueError(sampler)

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        for hid in hids:
            aggregators.append(SAGEAggregator(in_features, hid,
                                              concat=concat, bias=bias,
                                              aggr=aggr))

            if act == "IF":
                snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
            elif act == 'LIF':
                snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
            elif act == 'PLIF':
                snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(act)

            in_features = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.p = p
        self.pooling = nn.Linear(len(adj_evolve)*in_features, out_features)
        self.time_step=len(adj_evolve)

    def encode(self, nodes,graph):
        spikes = []
        sizes = self.sizes
        for time_step in range(len(graph)):
            snapshot = graph[time_step]
            sampler = self.sampler[time_step]
            sampler_t = self.sampler_t[time_step]

            x = snapshot.x
            h = [x[nodes]]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in sizes:
                size_1 = max(int(size * self.p), 1)
                size_2 = size - size_1
                if size_2 > 0:
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)

                num_nodes.append(nbr.size(0))
                h.append(x[nbr])

            for i, aggregator in enumerate(self.aggregators):

                self_x = h[:-1]
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))

                out = self.snn[i](aggregator(self_x, neigh_x))
                if i != len(sizes) - 1:
                    out = self.dropout(out)
                    h = torch.split(out, num_nodes[:-(i + 1)])

            spikes.append(out)

        spikes = torch.cat(spikes, dim=1)
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes,graphs):
        spikes= self.encode(nodes,graphs)
        if len(graphs)<self.time_step:
            pat=torch.zeros((graphs[-1].x.shape[0],int(spikes.shape[1]/len(graphs))*(self.time_step-len(graphs)))).to(spikes.device)
            spikes=torch.cat((spikes,pat),dim=1)
        spikes=self.pooling(spikes)
        return spikes