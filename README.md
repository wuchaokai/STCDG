# STCDG
a model

All configurations are reflected in the config/config.py file.

Prepare the similarity matrix :  #dtw(slow but accuracy)/pagerank

python link_prediction.py --dataset dblp --gen_sim True  --sim_type dtw

Run the model:

python link_prediction.py --dataset dblp --model EvolveGCN  --compensate

Run the base model:

python link_prediction.py --dataset dblp --model EvolveGCN



