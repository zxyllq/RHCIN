# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from os.path import join
import abc
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pdb
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, dice


__all__ = ["BaseModel"]


class BaseModel:
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.graph = graph if graph is not None else tf.Graph()
        self.iterator = iterator_creator(hparams, self.graph)
        self.train_num_ngs = (
            hparams.train_num_ngs if "train_num_ngs" in hparams else None
        )

        with self.graph.as_default():
            self.hparams = hparams

            self.layer_params = []
            self.embed_params = []
            self.cross_params = []
            self.layer_keeps = tf.placeholder(tf.float32, name="layer_keeps")
            self.keep_prob_train = None
            self.keep_prob_test = None
            self.is_train_stage = tf.placeholder(
                tf.bool, shape=(), name="is_training"
            )
            self.group = tf.placeholder(tf.int32, shape=(), name="group")

            self.initializer = self._get_initializer(self.seed)
            
            self.initializer_fuzhu = self._get_initializer_fuzhu(self.seed )

            self.logit  = self._build_graph()
            self.pred = self._get_pred(self.logit, self.hparams.method)

            self.loss = self._get_loss()
            self.saver = tf.train.Saver(max_to_keep=self.hparams.epochs)
            self.update = self._build_train_opt()
            self.extra_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS
            )
            self.init_op = tf.global_variables_initializer()
            self.merged = self._add_summaries()

        # set GPU use with demand growth
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(
            graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )
        
        self.sess.run(self.init_op)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
        pass

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.
        
        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)
        
        Returns:
            obj: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def _add_summaries(self):
        pass

    def _l2_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.embed_l2, tf.nn.l2_loss(param))
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.embed_l1, tf.norm(param, ord=1))
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.layer_l1, tf.norm(param, ord=1))
            )
        return l1_loss

    def _cross_l_loss(self):
        """Construct L1-norm and L2-norm on cross network parameters for loss function.
        Returns:
            obj: Regular loss value on cross network parameters.
        """
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l1, tf.norm(param, ord=1))
            )
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l2, tf.norm(param, ord=2))
            )
        return cross_l_loss

    def _get_initializer(self,seed):
        if self.hparams.init_method == "tnormal":
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed= seed
            )
        elif self.hparams.init_method == "uniform":
            return tf.random_uniform_initializer(
                -self.hparams.init_value, self.hparams.init_value, seed=  seed
            )
        elif self.hparams.init_method == "normal":
            return tf.random_normal_initializer(
                stddev=self.hparams.init_value, seed=  seed
            )
        elif self.hparams.init_method == "xavier_normal":
            return tf.contrib.layers.xavier_initializer(uniform=False, seed= seed)
        elif self.hparams.init_method == "xavier_uniform":
            return tf.contrib.layers.xavier_initializer(uniform=True, seed= seed)
        elif self.hparams.init_method == "he_normal":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=False, seed= seed
            )
        elif self.hparams.init_method == "he_uniform":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=True, seed= seed
            )
        else:
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=  seed
            )
    def _get_initializer_fuzhu(self,seed):
        
         
        return tf.contrib.layers.xavier_initializer(uniform=False, seed= seed)
         

    def _compute_data_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.logit, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "square_loss":
            data_loss = tf.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(
                        tf.reshape(self.pred, [-1]),
                        tf.reshape(self.iterator.labels, [-1]),
                    )
                )
            )
        elif self.hparams.loss == "log_loss":
            data_loss = tf.reduce_mean(
                tf.losses.log_loss(
                    predictions=tf.reshape(self.pred, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "softmax":
            
            group = self.train_num_ngs + 1
            logits = tf.reshape(self.logit, (-1, group))#<tf.Tensor 'Reshape:0' shape=(?, 5) dtype=float32>
            labels = tf.reshape(self.iterator.labels, (-1, group))#<tf.Tensor 'Reshape:0' shape=(?, 5) dtype=float32>
            softmax_pred = tf.nn.softmax(logits, axis=-1)#tf.Tensor 'Softmax:0' shape=(?, 5) dtype=float32>
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
             
            self.loss_each_sample = tf.math.log(tf.where(boolean_mask, softmax_pred, mask_paddings))
            data_loss = -group * tf.reduce_mean(self.loss_each_sample)
             


        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss #注意，只有softmax loss才可以 

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.
        Returns:
            obj: Regular loss.
        """
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        return tf.reduce_sum(regular_loss)

    def _train_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            obj: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adadelta":
            train_step = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == "adagrad":
            train_step = tf.train.AdagradOptimizer(lr)
        elif optimizer == "sgd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "adam":
            train_step = tf.train.AdamOptimizer(lr)
        elif optimizer == "ftrl":
            train_step = tf.train.FtrlOptimizer(lr)
        elif optimizer == "gd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "padagrad":
            train_step = tf.train.ProximalAdagradOptimizer(lr)
        elif optimizer == "pgd":
            train_step = tf.train.ProximalGradientDescentOptimizer(lr)
        elif optimizer == "rmsprop":
            train_step = tf.train.RMSPropOptimizer(lr)
        elif optimizer == "lazyadam":
            train_step = tf.contrib.opt.LazyAdamOptimizer(lr)
        else:
            train_step = tf.train.GradientDescentOptimizer(lr)
        return train_step

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()
        gradients, variables = zip(*train_step.compute_gradients(self.loss))
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return train_step.apply_gradients(zip(gradients, variables))

    def _active_layer(self, logit, activation, layer_idx=-1):
        """Transform the input value with an activation. May use dropout.
        
        Args:
            logit (obj): Input value.
            activation (str): A string indicating the type of activation function.
            layer_idx (int): Index of current layer. Used to retrieve corresponding parameters
        
        Returns:
            obj: A tensor after applying activation function on logit.
        """
        if layer_idx >= 0 and self.hparams.user_dropout:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        return self._activate(logit, activation, layer_idx)

    def _activate(self, logit, activation, layer_idx=-1):
        if activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.nn.softmax(logit)
        elif activation == "relu":
            return tf.nn.relu(logit)
        elif activation == "tanh":
            return tf.nn.tanh(logit)
        elif activation == "elu":
            return tf.nn.elu(logit)
        elif activation == "identity":
            return tf.identity(logit)
        elif activation == 'dice':
            return dice(logit, name='dice_{}'.format(layer_idx))
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, keep_prob):
        """Apply drops upon the input value.
        Args:
            logit (obj): The input value.
            keep_prob (float): The probability of keeping each element.

        Returns:
            obj: A tensor of the same shape of logit.
        """
        return tf.nn.dropout(x=logit, keep_prob=keep_prob)

     

     

     
    def load_model(self, model_path=None):
        """Load an existing model.

        Args:
            model_path: model path.

        Raises:
            IOError: if the restore operation failed.
        """
        act_path = self.hparams.load_saved_model
        if model_path is not None:
            act_path = model_path

        try:
            self.saver.restore(self.sess, act_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(act_path))

     

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.
        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.
        Returns:
            all_labels: labels after group.
            all_preds: preds after group.
        """
        all_keys = list(set(group_keys))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}
        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        return all_labels, all_preds

     
    def _attention_fcn(self, query, user_embedding,mask):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.variable_scope("attention_fcn",reuse=tf.AUTO_REUSE):
            query_size = query.shape[1].value
            boolean_mask = tf.equal(mask, tf.ones_like(mask))

            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
            )
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            return output
     

     

    def _fcn_net(self, model_output, layer_sizes, scope):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    # tf.summary.histogram(
                    #     "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                    # )
                    # tf.summary.histogram(
                    #     "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                    # )
                    curr_hidden_nn_layer = (
                        tf.tensordot(
                            hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                        )
                        + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(idx)
                    activation = hparams.activation[idx]

                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.get_variable(
                    name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
                )
                b_nn_output = tf.get_variable(
                    name="b_nn_output",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                # tf.summary.histogram(
                #     "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
                # )
                # tf.summary.histogram(
                #     "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
                # )
                nn_output = (
                    tf.tensordot(hidden_nn_layers[-1], w_nn_output, axes=1)
                    + b_nn_output
                )
               
                return nn_output

