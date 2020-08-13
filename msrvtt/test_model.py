# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-10-07
import tensorflow as tf
import pickle, json, h5py
import numpy as np
import sys, os
from pprint import pprint
from collections import defaultdict
import time
sys.path.append('..')
from utils2 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

np.random.seed(42)
data_dict = None
model = None
options = None

min_xe = 1000.

def cal_metrics(sess):
    sent_dict = defaultdict(list)
    loss_list = []

    tag_feat = data_dict['tag_feat']
    eco_feat = data_dict['eco_feat']
    res_feat = data_dict['res_feat']
    for idx in tag_feat:
        tag, evid, rvid = tag_feat[idx][:], eco_feat['avgpool'][idx][:]/32.973293, res_feat['avgpool'][idx][:]/10.586794
        eprob, rprob = eco_feat['probs'][idx][:][np.newaxis], res_feat['probs'][idx][:][np.newaxis]
        vid = np.concatenate([evid, rvid], axis=0)[np.newaxis]
        tag = np.concatenate([tag, eprob, rprob], axis=1)
        # print('tag shape:', tag.shape, 'evid:', evid.shape, 'rvid:', rvid.shape)
        wanted_ops = {'generated_words': model.generated_words}
        feed_dict = {model.vid_inputs: vid, model.se_inputs: tag, model.word_idx: np.ones(shape=(20, 1), dtype=np.int32)}
        # sel_word_idx shape: (batch_size, beam_width, n_steps)
        res = sess.run(wanted_ops, feed_dict)
        generated_words = res['generated_words']
        for x in np.squeeze(generated_words):
            if x == 0:
                break
            sent_dict[idx].append(data_dict['idx2word'][x])
        sent_dict[idx] = [' '.join(sent_dict[idx])]

    with open("./data/pretraining_prediction_model4.json", "w") as fo:
        json.dump(sent_dict, fo)


def main():
    global data_dict, model, options
    corpus = pickle.load(open("/home/chenhaoran/data2/msrvtt_dataset/msrvtt_corpus_cleaned_v2.pkl", "rb"))
    data_dict = {
    'tag_feat': h5py.File("/home/chenhaoran/data2/VideoCaptioner3/SCN/tagging/data/pretraining_test_semantics_msrvtt4.hdf5"),
    'eco_feat': h5py.File("/home/chenhaoran/data4/video_feats/test_eco_feats.hdf5", "r"),
    "res_feat": h5py.File("/home/chenhaoran/data4/video_feats/test_res_feats.hdf5", "r"),
    "corpus": corpus,
    "idx2word": corpus[4],
    }
    options = get_options(data_dict)
    model = get_model(options)
    # model = get_gru(options)
    best_score, save_path = 0., None

    with model.graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        train_op = get_train_op(model, options, global_step)
        saver = tf.train.Saver()
        config = get_config()
        sess = tf.Session(config=config, graph=model.graph)
        trainable_variables = tf.trainable_variables()
        # print(*trainable_variables, sep='\n', flush=True)
        saver.restore(sess, "./saves2/msrvtt_ref-best.ckpt")
        cal_metrics(sess)

        sess.close()


if __name__ == "__main__":

    flags = tf.app.flags.FLAGS

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    print('Total time: %.2fs' % (end_time - start_time))
