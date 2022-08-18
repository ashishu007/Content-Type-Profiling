# # top_k (-tk) here should be same as what was used in prev AL run
# python3 src/merge_annotated.py -not_first_run

# # top_k (-tk) here will be what we use in the next AL run
# CUDA_VISIBLE_DEVICES=7 python3 src/al_main.py --dataset mlb -qs qbc -tk 25 -do_al -a_class

CUDA_VISIBLE_DEVICES=7 python3 src/al_main.py --dataset mlb -qs clust -tk 25 -do_al -a_class
