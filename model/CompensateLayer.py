import torch
import numpy as np
import torch.nn as nn
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import torch.nn.functional as F
class CompensateLayer(nn.Module):
    def __init__(self,args,sim_list,evolve_adjs,TimeWeight,acF=nn.Sigmoid()):
        super(CompensateLayer, self).__init__()

        self.gcnlayer=GraphConvolution(args.nout,args.nout)
        self.sim_list=sim_list
        self.TimeWeight=TimeWeight
        self.gcnlayers=nn.ModuleList([
            GraphConvolution(args.nout,args.nout)
            for _ in range(args.time_steps)
        ])
        self.sim_temperalLayer=TemporalLayer(args.nout,args.nout,args.nout,1,args.time_steps,0.5,True,args.re_w)
        self.sim_temperalLayer2 = TemporalLayer2(args.nout, args.nout, args.nout, 1, args.time_steps, 0.5, True,
                                               args.re_w)
        #self.var_temperalLayer = VarTemporalLayer2(args.nout, args.nout, args.nout, 1, args.time_steps, 0.5, False)
        #self.var_temperalLayer=VarTemporalLayer(args.nout,args.nout,args.nout,args.time_steps,0.5,False)
        self.var_temperalLayer = VarTemporalLayer3(args.nout, args.nout, args.nout, args.time_steps, 0.5, True,args.trend_w)
        self.var_temperalLayer2 = VarTemporalLayer4(args.nout, args.nout, args.nout, args.time_steps, 0.5, True,
                                                   args.trend_w)
        #self.proj=nn.Sequential(MLPLayer(args.nout*2,args.nout*2,True,F.relu),MLPLayer(args.nout*2,args.nout*2,True))
        self.evolve_adjs=evolve_adjs
        #self.var_temperalLayer=
        self.acF=acF
        self.type=args.type
        self.args=args
        self.linear1 = nn.Linear(args.nout, args.nout,bias=False)
        self.linear2 = nn.Linear(args.nout, args.nout,bias=False)
        self.weight1=nn.Parameter(torch.ones((args.node_num,args.time_steps, args.nout)))
        self.weight2 = nn.Parameter(torch.ones((args.node_num,args.time_steps, args.nout)))
    def forward(self,feat_list,comb_type='concat'):
        new_feat_list=[]
        comb_type=self.args.comb
        for i in range(len(feat_list)):
            new_feat=self.gcnlayer(feat_list[i],self.sim_list[i])+feat_list[i]
            new_feat_list.append(new_feat)
        if self.args.without=='time':
            return new_feat_list
            #new_feat_list.append(torch.concat((self.gcnlayers[i](feat_list[i], self.sim_list[i]),feat_list[i]),dim=1))
        #time compensation
        elif self.args.without=='space':
            new_feat_list = torch.cat(
                (self.sim_temperalLayer(feat_list), self.var_temperalLayer(feat_list, self.TimeWeight)),
                dim=-1)
        elif self.args.without=='evo':
            new_feat_list = self.sim_temperalLayer(new_feat_list)
        elif self.args.without=='sim_time':
            new_feat_list = self.var_temperalLayer(new_feat_list,self.TimeWeight)
        else:
            if comb_type=='concat':
                new_feat_list=torch.cat((self.sim_temperalLayer(new_feat_list),self.var_temperalLayer(new_feat_list,self.TimeWeight)),dim=-1)
            elif comb_type=='mean':
                new_feat_list=0.5*self.sim_temperalLayer2(new_feat_list)+0.5*self.var_temperalLayer2(new_feat_list,self.TimeWeight)
                #new_feat_list=self.weight1[:feat_list[0].shape[0],:len(feat_list),:]*self.sim_temperalLayer(new_feat_list)+self.weight2[:feat_list[0].shape[0],:len(feat_list),:]*self.var_temperalLayer(new_feat_list,self.TimeWeight)
        new_feat_list=list(torch.unbind(new_feat_list, dim=1))
        return  new_feat_list

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input,adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TemporalLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 out_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual,weight):
        super(TemporalLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, hid_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, hid_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, hid_dim))
        # ff
        self.lin = nn.Linear(hid_dim, out_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()
        self.weight=weight
    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        stacked_tensor = torch.stack(inputs)
        inputs=stacked_tensor.permute(1, 0, 2)

        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, inputs.shape[1]).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs  +self.position_embeddings[position_inputs] # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # # 3: Split, concat and scale.
        # split_size = int(q.shape[-1] / self.n_heads)
        # q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        # k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        # v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q, k.permute(0, 2, 1))  # [hN, T, T]

        #outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)


        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)

        if self.residual:
            outputs = self.weight*outputs+temporal_inputs
        #return list(torch.unbind(outputs, dim=1))
        return outputs
    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)



class VarTemporalLayer3(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 output_dim,
                 num_time_steps,
                 attn_drop,
                 residual,weight):
        super(VarTemporalLayer3, self).__init__()
        self.num_time_steps = num_time_steps
        self.residual = residual


        self.gcnlayer = GraphConvolution(input_dim, output_dim)


        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, hid_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        # ff
        self.lin = nn.Linear(hid_dim, output_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.weight=weight
        self.xavier_init()

    def forward(self, inputs,TimeWeight):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        stacked_tensor = torch.stack(inputs)
        inputs=stacked_tensor.permute(1, 0, 2)
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, inputs.shape[1]).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs  +self.position_embeddings[position_inputs] # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))

        #outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        TimeWeight=TimeWeight[-v.shape[1]:,:,-v.shape[1]:]
        #TimeWeight=torch.stack(TimeWeight)
        TimeWeight=TimeWeight.permute(1,0,2)
        outputs = F.softmax(TimeWeight, dim=2)

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v)  # [hN, T, F/h]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)

        if self.residual:
            outputs = self.weight*outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.V_embedding_weights)
class TemporalLayer2(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 out_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual,weight):
        super(TemporalLayer2, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, hid_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, hid_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, hid_dim))
        # ff
        self.lin = nn.Linear(hid_dim, out_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()
        self.weight=weight
    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        stacked_tensor = torch.stack(inputs)
        inputs=stacked_tensor.permute(1, 0, 2)

        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, inputs.shape[1]).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs  +self.position_embeddings[position_inputs] # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # # 3: Split, concat and scale.
        # split_size = int(q.shape[-1] / self.n_heads)
        # q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        # k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        # v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q, k.permute(0, 2, 1))  # [hN, T, T]

        #outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)


        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)

        if self.residual:
            outputs = self.weight*outputs+inputs
        #return list(torch.unbind(outputs, dim=1))
        return outputs
    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)



class VarTemporalLayer4(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 output_dim,
                 num_time_steps,
                 attn_drop,
                 residual,weight):
        super(VarTemporalLayer4, self).__init__()
        self.num_time_steps = num_time_steps
        self.residual = residual


        self.gcnlayer = GraphConvolution(input_dim, output_dim)


        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, hid_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        # ff
        self.lin = nn.Linear(hid_dim, output_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.weight=weight
        self.xavier_init()

    def forward(self, inputs,TimeWeight):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        stacked_tensor = torch.stack(inputs)
        inputs=stacked_tensor.permute(1, 0, 2)
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, inputs.shape[1]).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs  +self.position_embeddings[position_inputs] # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))

        #outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        TimeWeight=TimeWeight[-v.shape[1]:,:,-v.shape[1]:]
        #TimeWeight=torch.stack(TimeWeight)
        TimeWeight=TimeWeight.permute(1,0,2)
        outputs = F.softmax(TimeWeight, dim=2)

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v)  # [hN, T, F/h]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)

        if self.residual:
            outputs = self.weight*outputs + inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.V_embedding_weights)
class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.activation is F.relu:
            gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats