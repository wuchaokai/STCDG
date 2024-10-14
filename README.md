# STCDG
a model

All configurations are reflected in the config/config.py file.

prepare the similarity matrix :  #dtw(slow but accuracy)/pagerank
python link_prediction.py --dataset dblp --gen_sim True  --sim_type dtw

run the model:
python link_prediction.py --dataset dblp --model EvolveGCN  --compensate

run the base model:
python link_prediction.py --dataset dblp --model EvolveGCN



