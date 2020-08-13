name=msrvtt_eco_res_avg_norm_semantic_tag_eco_efficientb7_corpus_cleaned_v3_ref_gamma_3_94_25
TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
CUDA_VISIBLE_DEVICES=8 \
python train_model.py --name $name \
    --corpus ../../video_feats/msrvtt_corpus_cleaned_v3.pkl \
    --ecores ../../video_feats/msrvtt_eco_res_avg_norm.npy \
    --tag    ../../video_feats/msrvtt_semantic_tag_eco_efficientb7_v3.npy \
    --ref    ../../video_feats/msrvtt_ref.pkl \
    --test   /home/chenhaoran/saves/
    > $name.log
