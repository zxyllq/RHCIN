# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import six
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import numpy as np
import pandas as pd
import yaml
import zipfile
import json
import pickle as pkl
import tensorflow as tf
import pdb
 


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.
    
    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def check_type(config):
    """Check that the config parameters are the correct type
    
    Args:
        config (dict): Configuration dictionary.

    Raises:
        TypeError: If the parameters are not the correct type.
    """

    int_parameters = [
        "word_size",
        "entity_size",
        "doc_size",
        "history_size",
        "FEATURE_COUNT",
        "FIELD_COUNT",
        "dim",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "PAIR_NUM",
        "DNN_FIELD_NUM",
        "attention_layer_sizes",
        "n_user",
        "n_item",
        "n_user_attr",
        "n_item_attr",
        "item_embedding_dim",
        "cate_embedding_dim",
        "user_embedding_dim",
        "max_seq_length",
        "hidden_size",
        "T",
        "L",
        "n_v",
        "n_h",
        "kernel_size",
        "min_seq_length",
        "attention_size",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "train_num_ngs",
        "clik_behavior",
        "max_session_count",
        "start_topN" ,
        "end_topN" ,
        "start_iteration"  ,                        
        "duration_iteration" ,                      
                                 
                                
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

    float_parameters = [
        "init_value",
        "learning_rate",
        "embed_l2",
        "embed_l1",
        "layer_l2",
        "layer_l1",
        "mu",
        "alpha_margin",
        "alpha_loss_weight"
    ]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("Parameters {0} must be float".format(param))

    str_parameters = [
        "train_file",
        "eval_file",
        "test_file",
        "infer_file",
        "method",
        "load_model_name",
        "infer_model_name",
        "loss",
        "optimizer",
        "init_method",
        "attention_activation",
        "user_vocab",
        "item_vocab",
        "cate_vocab",
        "data_type"
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = [
        "layer_sizes",
        "activation",
        "dropout",
        "att_fcn_layer_sizes",
        "dilations",
    ]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("Parameters {0} must be list".format(param))


def check_nn_config(f_config):
    """Check neural networks configuration.
    
    Args:
        f_config (dict): Neural network configuration.
    
    Raises:
        ValueError: If the parameters are not correct.
    """
     
    
    if f_config["model_type"] in ["ddsin" ]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "attention_size",
            "hidden_size",
            "att_fcn_layer_sizes",
            "discrepancy_loss_weight",
           # "contrastive_loss_weight",
            "is_clip_norm",
            #"data_type",
            "clik_behavior",
            "max_session_count"
            
        ]
     
    else:
        required_parameters = []

    # check required parameters
 
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameters {0} must be set".format(param))

     
    check_type(f_config)


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def create_hparams(flags):
    """Create the model hyperparameters.

    Args:
        flags (dict): Dictionary with the model requirements.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    return tf.contrib.training.HParams(
        # data
      ##############################################################################
        clik_behavior=flags["clik_behavior"] if "clik_behavior" in flags else None,
        alpha_margin =flags[ "alpha_margin"] if"alpha_margin" in flags else None ,
        alpha_loss_weight = flags["alpha_loss_weight"]  if "alpha_loss_weight" in flags else None,
        max_session_count = flags["max_session_count"] if "max_session_count" in flags else None,
        start_topN = flags["start_topN"] if "start_topN" in flags else None,
        end_topN = flags["end_topN"] if "end_topN" in flags else None,
        start_iteration =flags["start_iteration"] if "start_iteration"  in flags else None,                      
        duration_iteration =flags["duration_iteration"] if "duration_iteration" in flags else None,
        max_seq_length =flags["max_seq_length"] if "max_seq_length" in flags else None,
        attention_size  =flags["attention_size"] if "attention_size" in flags else None,
        valid_step=flags["valid_step"] if "valid_step" in flags else None,
        start_forget_lambda = flags["start_forget_lambda"] if "start_forget_lambda" in flags else None,
        end_forget_lambda = flags["end_forget_lambda"] if "end_forget_lambda" in flags else None,
        model = flags["model"] if "model" in flags else None,
        tau = flags["tau"] if "tau" in flags else 1.0,
        augment_rate=flags["augment_rate"] if "augment_rate" in flags else 0.2,
        alpha_fcn_layer_sizes =flags["alpha_fcn_layer_sizes"] if "alpha_fcn_layer_sizes" in flags else None,
     ########################################################
        att_fcn_layer_sizes=flags["att_fcn_layer_sizes"] if "att_fcn_layer_sizes" in flags else None,
        hidden_size=flags["hidden_size"] if "hidden_size" in flags else None,
        max_grad_norm=flags["max_grad_norm"] if "max_grad_norm" in flags else 2,
        SUMMARIES_DIR=flags["SUMMARIES_DIR"] if "SUMMARIES_DIR" in flags else None,
        MODEL_DIR=flags["MODEL_DIR"] if "MODEL_DIR" in flags else None,
        
        
        # model
        dim=flags["dim"] if "dim" in flags else None,
        layer_sizes=flags["layer_sizes"] if "layer_sizes" in flags else None,
       
      
        activation=flags["activation"] if "activation" in flags else None,
      
       
        user_dropout=flags["user_dropout"] if "user_dropout" in flags else False,
        dropout=flags["dropout"] if "dropout" in flags else [0.0],
        attention_layer_sizes=flags["attention_layer_sizes"]
        if "attention_layer_sizes" in flags
        else None,
        attention_activation=flags["attention_activation"]
        if "attention_activation" in flags
        else None,
        attention_dropout=flags["attention_dropout"]
        if "attention_dropout" in flags
        else 0.0,
        model_type=flags["model_type"] if "model_type" in flags else None,
        method=flags["method"] if "method" in flags else None,
        load_saved_model=flags["load_saved_model"]
        if "load_saved_model" in flags
        else False,
        load_model_name=flags["load_model_name"]
        if "load_model_name" in flags
        else None,
        
       
         
        # train
        init_method=flags["init_method"] if "init_method" in flags else "tnormal",
        init_value=flags["init_value"] if "init_value" in flags else 0.01,
        embed_l2=flags["embed_l2"] if "embed_l2" in flags else 0.0000,
        embed_l1=flags["embed_l1"] if "embed_l1" in flags else 0.0000,
        layer_l2=flags["layer_l2"] if "layer_l2" in flags else 0.0000,
        layer_l1=flags["layer_l1"] if "layer_l1" in flags else 0.0000,
        cross_l2=flags["cross_l2"] if "cross_l2" in flags else 0.0000,
        cross_l1=flags["cross_l1"] if "cross_l1" in flags else 0.0000,
        attn_loss_weight=flags["attn_loss_weight"] if "attn_loss_weight" in flags else 0.0000,
         
        discrepancy_loss_weight=flags["discrepancy_loss_weight"] if "discrepancy_loss_weight" in flags else 0.0000,
        contrastive_loss_weight=flags["contrastive_loss_weight"] if "contrastive_loss_weight" in flags else 0.0000,
        contrastive_length_threshold=flags["contrastive_length_threshold"] if "contrastive_length_threshold" in flags else 1,
        contrastive_recent_k=flags["contrastive_recent_k"] if "contrastive_recent_k" in flags else 3,
        triplet_margin=flags["triplet_margin"] if "triplet_margin" in flags else 1.0,
        contrastive_loss=flags["contrastive_loss"] if "contrastive_loss" in flags else "bpr",
        learning_rate=flags["learning_rate"] if "learning_rate" in flags else 0.001,

        
        
        is_clip_norm=flags["is_clip_norm"] if "is_clip_norm" in flags else 0,
       
        dtype=flags["dtype"] if "dtype" in flags else 32,
        loss=flags["loss"] if "loss" in flags else None,
        optimizer=flags["optimizer"] if "optimizer" in flags else "adam",
        epochs=flags["epochs"] if "epochs" in flags else 10,
        batch_size=flags["batch_size"] if "batch_size" in flags else 1,
        enable_BN=flags["enable_BN"] if "enable_BN" in flags else False,
        # show info
        show_step=flags["show_step"] if "show_step" in flags else 1,
        save_model=flags["save_model"] if "save_model" in flags else True,
        save_epoch=flags["save_epoch"] if "save_epoch" in flags else 5,
        metrics=flags["metrics"] if "metrics" in flags else None,
        write_tfevents=flags["write_tfevents"] if "write_tfevents" in flags else False,
        # sequential
        item_embedding_dim=flags["item_embedding_dim"]
        if "item_embedding_dim" in flags
        else None,
        cate_embedding_dim=flags["cate_embedding_dim"]
        if "cate_embedding_dim" in flags
        else None,
        user_embedding_dim=flags["user_embedding_dim"]
        if "user_embedding_dim" in flags
        else None,
        train_num_ngs=flags["train_num_ngs"] if "train_num_ngs" in flags else 4,
        need_sample=flags["need_sample"] if "need_sample" in flags else True,
        embedding_dropout=flags["embedding_dropout"]
        if "embedding_dropout" in flags
        else 0.3,
        user_vocab=flags["user_vocab"] if "user_vocab" in flags else None,
        item_vocab=flags["item_vocab"] if "item_vocab" in flags else None,
        cate_vocab=flags["cate_vocab"] if "cate_vocab" in flags else None,
        pairwise_metrics=flags["pairwise_metrics"]
        if "pairwise_metrics" in flags
        else None,
        weighted_metrics=flags["weighted_metrics"]
        if "weighted_metrics" in flags
        else None,
        EARLY_STOP=flags["EARLY_STOP"] if "EARLY_STOP" in flags else 100,
        
         
    )


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the model hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    if kwargs:
        for name, value in six.iteritems(kwargs):
            config[name] = value

    check_nn_config(config)
    return create_hparams(config)


 


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    
    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    if not metrics:
        return res

    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def cal_weighted_metric(users, preds, labels, metrics):

    res = {}
    if not metrics:
        return res

    df = pd.DataFrame({'users': users, 'preds': preds, 'labels': labels})
    weight = df[["users", "labels"]].groupby("users").count().reset_index().set_index("users", drop=True).rename(columns={"labels": "weight"})
    weight["weight"] = weight["weight"]/weight["weight"].sum()
    for metric in metrics:
        if metric == 'wauc':
            wauc = cal_wauc(df, weight)
            res["wauc"] = round(wauc, 4)
        elif metric == 'wmrr':
            wmrr = cal_wmrr(df, weight)
            res["wmrr"] = round(wmrr, 4)
        elif metric.startswith("whit"): # format like: whit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            whit_res = cal_whit(df, weight, hit_list)
            res.update(whit_res)
        elif metric.startswith("wndcg"): # format like: wndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            wndcg_res = cal_wndcg(df, weight, ndcg_list)
            res.update(wndcg_res)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def cal_wauc(df, weight):

    weight["auc"] = df.groupby("users").apply(groupby_auc)
    wauc_score = (weight["weight"]*weight["auc"]).sum()
    weight.drop(columns="auc", inplace=True)

    return wauc_score


def groupby_auc(df):

    y_hat = df.preds
    y = df.labels
    return roc_auc_score(y, y_hat)


def cal_wmrr(df, weight):

    weight["mrr"] = df.groupby("users").apply(groupby_mrr)
    wmrr_score = (weight["weight"]*weight["mrr"]).sum()
    weight.drop(columns="mrr", inplace=True)

    return wmrr_score


def groupby_mrr(df):

    y_hat = df.preds
    y = df.labels
    return mrr_score(y, y_hat)


def cal_whit(df, weight, hit_list):

    whit_res = {}
    weight["hit"] = df.groupby("users").apply(groupby_hit, hit_list=hit_list)
    whit_score = (weight["weight"]*weight["hit"]).sum()
    weight.drop(columns="hit", inplace=True)
    for i, k in enumerate(hit_list):
        metric = "whit@{0}".format(k)
        whit_res[metric] = round(whit_score[i], 4)

    return whit_res


def groupby_hit(df, hit_list):

    y_hat = df.preds
    y = df.labels
    hit = np.array([hit_score(y, y_hat, k) for k in hit_list])

    return hit


def cal_wndcg(df, weight, ndcg_list):

    wndcg_res = {}
    weight["ndcg"] = df.groupby("users").apply(groupby_ndcg, ndcg_list=ndcg_list)
    wndcg_score = (weight["weight"]*weight["ndcg"]).sum()
    weight.drop(columns="ndcg", inplace=True)
    for i, k in enumerate(ndcg_list):
        metric = "wndcg@{0}".format(k)
        wndcg_res[metric] = round(wndcg_score[i], 4)

    return wndcg_res


def groupby_ndcg(df, ndcg_list):

    y_hat = df.preds
    y = df.labels
    ndcg = np.array([ndcg_score(y, y_hat, k) for k in ndcg_list])

    return ndcg


def cal_mean_alpha_metric(alphas, labels):

    res = {}

    alphas = np.asarray(alphas)
    labels = np.asarray(labels)
    res["mean_alpha"] = round((alphas*labels).sum()/labels.sum(), 4)

    return res


def load_dict(filename):
    """Load the vocabularies.

    Args:
        filename (str): Filename of user, item or category vocabulary.

    Returns:
        dict: A saved vocabulary.
    """
    with open(filename, "rb") as f:
        f_pkl = pkl.load(f)
        return f_pkl


def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) +
                        epsilon, axis=reduction_axes)
    std = tf.math.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x
