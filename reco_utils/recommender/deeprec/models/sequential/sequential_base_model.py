# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import socket
import pdb 
from reco_utils.recommender.deeprec.models.base_model import BaseModel
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, load_dict

__all__ = ["SequentialBaseModel"]


class SequentialBaseModel(BaseModel):
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams

        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams else 1
        )
        self.hidden_size = hparams.hidden_size if "hidden_size" in hparams else None
        self.graph = tf.Graph() if not graph else graph

        with self.graph.as_default():
            self.embedding_keeps = tf.placeholder(tf.float32, name="embedding_keeps")
            self.embedding_keep_prob_train = None
            self.embedding_keep_prob_test = None

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass

    def _build_graph(self):
        """The main function to create sequential models.
        
        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        self.embedding_keep_prob_test = 1.0

        with tf.variable_scope("sequential") as self.sequential_scope:
            
            self._build_embedding()
            self._lookup_from_embedding()
            
            
            if  hparams.model in ["DDSIN_7"]:
                model_output ,model_output_aux = self._build_seq_graph()
                logit  = self._fcn_net( model_output, hparams.layer_sizes, scope="logit_fcn")
                self.logit_aux = self._fcn_net(  model_output_aux , hparams.layer_sizes, scope="logit_fcn")
            else:
                model_output  = self._build_seq_graph()
                logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            
                 
            self._add_norm()
            return logit

    def train(self, sess, feed_dict):

        pass

    def batch_train(self, file_iterator, train_sess):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        pass

    def fit(
        self, train_file, valid_file, valid_num_ngs, eval_metric="group_auc" 
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            obj: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        self.eval_info = list()

        self.best_metric, self.best_epoch = 0, 0
        self.all_step=0
        self.valid_stop_patience=0
        for epoch in range(1, self.hparams.epochs + 1):
            self.hparams.current_epoch = epoch
            file_iterator = self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            
            epoch_loss = self.batch_train(file_iterator, train_sess, valid_file,valid_num_ngs, epoch,eval_metric )
           
            if self.hparams.EARLY_STOP > 0 and self.valid_stop_patience >= self.hparams.EARLY_STOP:
                print("early stop at epoch {0} !".format(epoch ))
                break 

             

        if self.hparams.write_tfevents:
            self.writer.close()

        print(self.eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self

     

    def eval(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).eval(sess, feed_dict)

    def run_weighted_eval(self, filename, num_ngs, calc_mean_alpha=False, manual_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            if manual_alpha:
                alphas = alphas[0]
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res
     

    def eval_with_user(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels], feed_dict=feed_dict)

    def eval_with_user_and_alpha(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels, self.alpha_output ], feed_dict=feed_dict)

     

     

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        
        

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        hparams = self.hparams
         

    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
       
        all_variables, embed_variables = (
            tf.trainable_variables(),
            tf.trainable_variables(self.sequential_scope._name + "/embedding"),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        self.layer_params.extend(layer_params)
