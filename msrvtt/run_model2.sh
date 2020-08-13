name=msrvtt_ref_cleaned_v2_ref_complete
TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
CUDA_VISIBLE_DEVICES=8 \
python train_model2.py --name $name \
    --corpus ../../msrvtt_dataset/msrvtt_corpus_cleaned_v2.pkl \
    --ecores ../../msrvtt_dataset/msrvtt_eco_res_avg_norm.npy \
    --tag    ../../msrvtt_dataset/msrvtt_semantic_tag_eco_res_avg_cleaned_v2.npy \
    --ref    ../../msrvtt_dataset/msrvtt_ref_v3_complete.pkl \
    --test   ./saves2/msrvtt_ref-best.ckpt
    > ./log2/${name}.log
