import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="DBLP",
                    help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+',
                    default=[32, 10], help='Hidden units for each layer. (default: [128, 10])')
parser.add_argument("--aggr", nargs="?", default="mean",
                    help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage",
                    help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
parser.add_argument("--surrogate", nargs="?", default="sigmoid",
                    help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="LIF",
                    help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size for training. (default: 1024)')
parser.add_argument('--compensate',action='store_true', default=False,help='True if compensating based model')
parser.add_argument('--without',type=str,default='space')
parser.add_argument('--model',type=str,default='space')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--train_size', type=float, default=0.4,
                    help='Ratio of nodes for training. (default: 0.4)')
parser.add_argument('--egcn_type', type=str, default='EGCNH', help='Type of EGCN: EGCNH or EGCNO')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--p', type=float, default=0.5,
                    help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout probability. (default: 0.7)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for model. (default: 2022)')
parser.add_argument('--select_rate',type=float,default=0.5)
parser.add_argument('--match_rate',type=float,default=0.5)
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--limit_time',type=int,default=0)
parser.add_argument('--rewiring_type', type=str, default='base', help='rewiring_type')
parser.add_argument('--gen_sim', type=bool, default=False, help='whether generate the space similar matrix')
parser.add_argument('--sim_type',type=str,default='dtw', help='dtw/pagerank')
args = parser.parse_args()
