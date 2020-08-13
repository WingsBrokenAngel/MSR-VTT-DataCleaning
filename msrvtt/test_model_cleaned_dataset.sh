name=msrvtt_ref_cleaned_ref
TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
CUDA_VISIBLE_DEVICES=8 \
python train_model.py --name $name \
    --corpus ../../msrvtt_feats/msrvtt_corpus_cleaned.pkl \
    --ecores ../../msrvtt_feats/msrvtt_eco_res_avg_norm.npy \
    --tag    ../../msrvtt_feats/msrvtt_semantic_tag_eco_res_avg_cleaned.npy \
    --ref    ../../msrvtt_feats/msrvtt_ref_cleaned.pkl \
    --test   ./saves/msrvtt_ref-best.ckpt
