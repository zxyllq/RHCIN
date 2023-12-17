# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.nn import dynamic_rnn
 
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

from deepctr.layers.sequence import  BiLSTM
import os
import numpy as np
from deepctr.layers.utils import concat_func
from deepctr.layers.sequence import  AttentionSequencePoolingLayer
import pdb 
from tensorflow.python.keras.layers import   Lambda
import math 
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, load_dict
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)
__all__ = ["RHCINModel"]

 
 

class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization using mean and variance
    gamma and beta are the learnable parameters
    """

    def __init__(self, seq_max_len, embedding_dim, epsilon):
        """Initialize parameters.

        Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            epsilon (float): Epsilon value.
        """
        super(LayerNormalization, self).__init__()
        
        self.epsilon = epsilon
        self.params_shape = ( seq_max_len,  embedding_dim)
        g_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=g_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=b_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )

    def call(self, x):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        mean, variance = tf.nn.moments(x, [-1], keepdims=True)
        normalized = (x - mean) / ((variance + self.epsilon) ** 0.5)
        output = self.gamma * normalized + self.beta
        return output

class RHCINModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
         
        
        self.graph = tf.Graph()  
        with self.graph.as_default():
            
            self.layer_normalization_short = LayerNormalization(
            2 *  hparams.max_seq_length,  2*hparams.attention_size, 1e-08
        ) 
            self.layer_normalization_long = LayerNormalization(
            hparams.max_seq_length  ,   hparams.attention_size, 1e-08
        ) 
             
        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)
        


    def _get_loss(self):
        """Make loss function, consists of data loss, regularization loss, contrastive loss and discrepancy loss
        
        Returns:
            obj: Loss value
        """
       
        self.forward_data_loss = self._compute_data_loss( )
        self.regular_loss = self._compute_regular_loss()
        self.discrepancy_loss = self._compute_discrepancy_loss()
        self.contrastive_loss = self._compute_contrastive_loss()

        self.loss =  self.forward_data_loss+  self.regular_loss  +self.discrepancy_loss +self.contrastive_loss
        return self.loss
    
    
    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """

        return super(RHCINModel, self)._build_train_opt()
    def _compute_discrepancy_loss(self):
        """Discrepancy loss between long and short term user embeddings."""
        discrepancy_loss = tf.reduce_mean(
            tf.math.squared_difference(
                tf.reshape(self.involved_user_long_embedding, [-1]),
                tf.reshape(self.involved_user_short_embedding, [-1])
            )
        )
        discrepancy_loss = -tf.multiply(self.hparams.discrepancy_loss_weight, discrepancy_loss)
        return discrepancy_loss
     
    def _compute_contrastive_loss(self):
        """Contrative loss on long and short term intention."""
        contrastive_mask_pre_1 = tf.where(
            tf.greater(self.current_session_pre_1_length , self.hparams.contrastive_length_threshold),
            tf.ones_like(self.current_session_pre_1_length , dtype=tf.bool),
            tf.zeros_like(self.current_session_pre_1_length , dtype=tf.bool)
        ) 
        contrastive_mask_pre_2 = tf.where(
            tf.greater(self.current_session_pre_2_length , self.hparams.contrastive_length_threshold),
            tf.ones_like(self.current_session_pre_2_length , dtype=tf.bool),
            tf.zeros_like(self.current_session_pre_2_length , dtype=tf.bool)
        )
         
        contrastive_mask =  tf.logical_and(contrastive_mask_pre_1, contrastive_mask_pre_2) #<tf.Tensor 'LogicalAnd_1:0' shape=(?,) dtype=bool>
        filtered_current_session_pre_fea_1 = tf.boolean_mask(self.att_fea_current_pre_rnn_1 , contrastive_mask )
        filtered_current_session_pre_fea_2 = tf.boolean_mask(self.att_fea_current_pre_rnn_2, contrastive_mask )
         
        filtered_current_session_pre_fea_1 = tf.reshape(filtered_current_session_pre_fea_1,[-1,self.hparams.train_num_ngs+1,self.att_fea_current_pre_rnn_1.shape[-1]])[:,0] 
        filtered_current_session_pre_fea_2 = tf.reshape(filtered_current_session_pre_fea_2,[-1,self.hparams.train_num_ngs+1,self.att_fea_current_pre_rnn_2.shape[-1]])[:,0] 

        filtered_current_session_pre_fea_1  =tf.nn.l2_normalize( filtered_current_session_pre_fea_1 , 1)
        filtered_current_session_pre_fea_2 =tf.nn.l2_normalize(filtered_current_session_pre_fea_2,1)
        all_score   = tf.matmul(filtered_current_session_pre_fea_1 ,tf.transpose(  filtered_current_session_pre_fea_2))#/ self.hparams.tau#<tf.Tensor 'MatMul_1:0' shape=(?, ?) dtype=float32>
        pos_score  = tf.diag_part(all_score )#<tf.Tensor 'DiagPart:0' shape=(?,) dtype=float32>
        contrastive_loss  = -(self.hparams.train_num_ngs+1)*tf.reduce_mean(tf.math.log( tf.exp(pos_score /self.hparams.tau)  / (tf.reduce_sum(tf.exp( all_score /self.hparams.tau),1 )+tf.reduce_sum(tf.exp( all_score /self.hparams.tau),0 )-tf.exp(pos_score /self.hparams.tau)) ),0) 

        contrastive_loss = tf.multiply(self.hparams.contrastive_loss_weight, contrastive_loss )
        return contrastive_loss
            
     
     



    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim

        with tf.variable_scope("embedding", initializer=self.initializer):
          
            self.item_lookup = tf.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup = tf.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
            self.user_long_lookup = tf.get_variable(
                name="user_long_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.user_short_lookup = tf.get_variable(
                name="user_short_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.position_embedding_lookup_long = tf.get_variable(
                name="position_embedding_long",
                shape=[self.hparams.max_seq_length, self.item_embedding_dim + self.cate_embedding_dim],
                dtype=tf.float32,
            )#加上position
            
            
             

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        #super(MyModel, self)._lookup_from_embedding()
        self.user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.iterator.users
        )
       # tf.summary.histogram("user_long_embedding_output", self.user_long_embedding)

        self.user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.iterator.users
        )
       # tf.summary.histogram("user_short_embedding_output", self.user_short_embedding)

        involved_users = tf.reshape(self.iterator.users, [-1])
        self.involved_users, _ = tf.unique(involved_users)
        self.involved_user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_long_embedding)
        self.involved_user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_short_embedding)


        self.item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.items
        )
        

        self.cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.cates
        )

        self.item_sequence_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_sequence
        )

        self.cate_sequence_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_sequence
        )
        self.item_sequence_pre_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_sequence_pre
        )

        self.cate_sequence_pre_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_sequence_pre
        )
####################################################################
        self.item_current_session_pre_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_current_session_pre
        )

        self.cate_current_session_pre_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_current_session_pre
        )

        self.item_current_session_pre_1_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_current_session_pre_1
        )

        self.cate_current_session_pre_1_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_current_session_pre_1
        )


        self.item_current_session_pre_2_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_current_session_pre_2
        )

        self.cate_current_session_pre_2_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_current_session_pre_2
        )
         

####################################################################


        self.item_session_0_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_session_0
        )

        self.cate_session_0_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_session_0
        )

        
        self.item_session_1_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_session_1
        )
        self.cate_session_1_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_session_1
        )
        self.item_session_2_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_session_2
        )
        self.cate_session_2_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_session_2
        )
        self.item_session_3_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_session_3
        )
        self.cate_session_3_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_session_3
        )
        self.item_session_4_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_session_4
        )
        self.cate_session_4_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_session_4
        )
        
        


        involved_items = tf.concat(
            [   tf.reshape(self.iterator.item_sequence, [-1]),
                tf.reshape(self.iterator.item_current_session_pre, [-1]),
                tf.reshape(self.iterator.item_current_session_pre_1, [-1]), 
                tf.reshape(self.iterator.item_current_session_pre_2, [-1]), 
                
                 
                

                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.involved_items
        )
        self.embed_params.append(involved_item_embedding)
        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_sequence, [-1]),
               tf.reshape(self.iterator.item_cate_current_session_pre, [-1]),
               tf.reshape(self.iterator.item_cate_current_session_pre_1, [-1]),
                tf.reshape(self.iterator.item_cate_current_session_pre_2, [-1]),
              
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)

        self.target_item_embedding = tf.concat(
            [self.item_embedding, self.cate_embedding], -1
        )
        self.position_embedding_long = tf.nn.embedding_lookup(
            self.position_embedding_lookup_long,
            tf.tile(tf.expand_dims(tf.range(self.hparams.max_seq_length), 0),
                    [ tf.shape( self.iterator.item_session_0 )[0] , 1])
        )
         
        self.embed_params.append(self.position_embedding_long)





        # dropout after embedding
        self.user_long_embedding = self._dropout(
            self.user_long_embedding, keep_prob=self.embedding_keeps
        )
        self.user_short_embedding = self._dropout(
            self.user_short_embedding, keep_prob=self.embedding_keeps
        )
        self.item_embedding = self._dropout(
            self.item_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_embedding = self._dropout(
            self.cate_embedding, keep_prob=self.embedding_keeps
        )

        self.item_sequence_embedding = self._dropout(
            self.item_sequence_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_sequence_embedding = self._dropout(
            self.cate_sequence_embedding, keep_prob=self.embedding_keeps
        )
        self.item_sequence_pre_embedding = self._dropout(
            self.item_sequence_pre_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_sequence_pre_embedding = self._dropout(
            self.cate_sequence_pre_embedding, keep_prob=self.embedding_keeps
        )
#######################################################################
        self.item_current_session_pre_embedding = self._dropout(
            self.item_current_session_pre_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_current_session_pre_embedding = self._dropout(
            self.cate_current_session_pre_embedding, keep_prob=self.embedding_keeps
        )
        self.item_current_session_pre_1_embedding = self._dropout(
            self.item_current_session_pre_1_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_current_session_pre_1_embedding = self._dropout(
            self.cate_current_session_pre_1_embedding, keep_prob=self.embedding_keeps
        )


        self.item_current_session_pre_2_embedding = self._dropout(
            self.item_current_session_pre_2_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_current_session_pre_2_embedding = self._dropout(
            self.cate_current_session_pre_2_embedding, keep_prob=self.embedding_keeps
        )
          
#######################################################################

        self.item_session_0_embedding = self._dropout(
            self.item_session_0_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_session_0_embedding = self._dropout(
            self.cate_session_0_embedding, keep_prob=self.embedding_keeps
        )

        

        self.item_session_1_embedding = self._dropout(
            self.item_session_1_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_session_1_embedding = self._dropout(
            self.cate_session_1_embedding, keep_prob=self.embedding_keeps
        )
        self.item_session_2_embedding = self._dropout(
            self.item_session_2_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_session_2_embedding = self._dropout(
            self.cate_session_2_embedding, keep_prob=self.embedding_keeps
        )
        self.item_session_3_embedding = self._dropout(
            self.item_session_3_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_session_3_embedding = self._dropout(
            self.cate_session_3_embedding, keep_prob=self.embedding_keeps
        )
        self.item_session_3_embedding = self._dropout(
            self.item_session_3_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_session_4_embedding = self._dropout(
            self.cate_session_4_embedding, keep_prob=self.embedding_keeps
        )


        self.target_item_embedding = self._dropout(
            self.target_item_embedding, keep_prob=self.embedding_keeps
        )
        self.position_embedding_long = self._dropout(
            self.position_embedding_long, keep_prob=self.embedding_keeps
        )
        


    def feedforward(self, inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    # dropout_rate=0.2,
                    # is_training=True,
                    reuse=None,mask =None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = self._dropout(outputs, keep_prob=self.embedding_keeps)
            # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            outputs = self._dropout(outputs, keep_prob=self.embedding_keeps)
            # Residual connection
             
            outputs += inputs
             

            # masking
            out2 = outputs  *  tf.expand_dims(mask,-1)

             

        return out2
    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=4,
                           
                             
                            scope="multihead_attention",
                            reuse=None,causality=False,
                           ):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

          
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            # key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k) (?, 20)
            key_masks =  tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) 
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k) 重复head次
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1,  queries.get_shape().as_list()[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if  causality ==True :
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)下三角矩阵
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)再次进行softmax

            # Query Masking, query_masks (N, T_q)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
            query_masks = tf.tile(query_masks, [ num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)


            # Dropouts
            outputs = self._dropout(outputs, keep_prob=self.embedding_keeps)
            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            
            outputs += queries

        return outputs

    def _build_sasrec(self,hist_input,mask,sequence_length,scope, num_heads,causality=False):
        hist_input *=tf.expand_dims(mask,-1)
         
        for i in range(2):
            with tf.variable_scope(scope+"_num_blocks_%d" % i,reuse=tf.AUTO_REUSE):

                # Self-attention
                if 'short' in scope :
                    queries=self.layer_normalization_short(hist_input)
                    # if i ==0 :
                        
                    #     queries = queries* tf.expand_dims(self.target_item_embedding,1)#交互式自注意力
                else :
                    queries=self.layer_normalization_long(hist_input)
                    # if i ==0 :
                        
                    #     queries = queries* tf.expand_dims(self.user_embedding,1)
                hist_input = self.multihead_attention(queries ,
                                               keys=hist_input,
                                               num_units= self.item_embedding_dim + self.cate_embedding_dim ,
                                               num_heads=num_heads,scope="self_attention" , causality=causality )


                # Feed forward
                if 'short' in scope :
                    inputs=self.layer_normalization_short(hist_input)
                else :
                    inputs=self.layer_normalization_long(hist_input)
                hist_input = self.feedforward(inputs , num_units=[ self.item_embedding_dim + self.cate_embedding_dim ,
                                                               self.item_embedding_dim + self.cate_embedding_dim ], mask = mask)
                
                
                # self.seq *= mask

       
        if 'short' in scope :
            hist_input=self.layer_normalization_short(hist_input)
        else :
            hist_input=self.layer_normalization_long(hist_input)
        if causality==True :
            right = tf.cast( sequence_length - tf.ones_like( sequence_length), tf.int32)
            right = tf.expand_dims(right, 1)
            left = tf.range(tf.shape( sequence_length)[0])
            left = tf.expand_dims(left, 1)
            ind_tensor = tf.concat([left, right], -1)
            final_state = tf.gather_nd(hist_input, ind_tensor)
        else:
            final_state=hist_input
        
        
        return final_state 


    def _build_seq_graph(self):
        """The main function to create clsr model.
        
        Returns:
            obj:the output of clsr section.
        """
        
        hparams = self.hparams

        with tf.variable_scope("rhsin"):
               
                #sequence
                sequence_input  = tf.concat(
                    [self.item_sequence_embedding, self.cate_sequence_embedding] ,  2
                )#(?,20,40)
                sequence_pre_input  = tf.concat(
                    [self.item_sequence_pre_embedding, self.cate_sequence_pre_embedding] ,  2
                ) 
                
                sequence_input=sequence_input[:,20:  ,:]
                mask_sequence = self.iterator.sequence_mask[:,20:   ]#(?, 20)
                #mask_sequence = self.iterator.sequence_mask
                real_mask_sequence = tf.cast( mask_sequence, tf.float32) 
                sequence_length = tf.reduce_sum(mask_sequence, 1)
                
                #current_session 
                current_session_pre_input   =tf.concat([self.item_current_session_pre_embedding ,self.cate_current_session_pre_embedding] ,2)
                #current_session_pre_input   = self.item_current_session_pre_embedding  
                mask_current_session_pre   =  self.iterator.current_session_pre_mask 
                real_mask_current_session_pre = tf.cast( mask_current_session_pre, tf.float32) 
                self.current_session_pre_length = tf.reduce_sum(mask_current_session_pre, 1)

                current_session_pre_1_input   =tf.concat([self.item_current_session_pre_1_embedding ,self.cate_current_session_pre_1_embedding] ,2)
                #current_session_pre_input   = self.item_current_session_pre_embedding  
                mask_current_session_pre_1   =  self.iterator.current_session_pre_mask_1 
                real_mask_current_session_pre_1 = tf.cast( mask_current_session_pre_1, tf.float32) 
                self.current_session_pre_1_length = tf.reduce_sum(mask_current_session_pre_1, 1)


                current_session_pre_2_input   =tf.concat([self.item_current_session_pre_2_embedding ,self.cate_current_session_pre_2_embedding] ,2)
                #current_session_pre_input   = self.item_current_session_pre_embedding  
                mask_current_session_pre_2   =  self.iterator.current_session_pre_mask_2 
                real_mask_current_session_pre_2 = tf.cast( mask_current_session_pre_2, tf.float32) 
                self.current_session_pre_2_length = tf.reduce_sum(mask_current_session_pre_2, 1)



                # session_0
                session_input_0 = tf.concat(
                    [self.item_session_0_embedding, self.cate_session_0_embedding ], 2
                )#(?,20,40)
                mask_0 = self.iterator.session_0_mask#(?, 20)
                real_mask_0 = tf.cast( mask_0, tf.float32)
                session_length_0 = tf.reduce_sum(mask_0, 1)
                 
                session_input_1 = tf.concat(
                    [self.item_session_1_embedding, self.cate_session_1_embedding ], 2
                )#(?,20,40)
                 # session_1
                mask_1 = self.iterator.session_1_mask#(?, 20)
                real_mask_1 = tf.cast( mask_1, tf.float32)
                session_length_1 = tf.reduce_sum( mask_1, 1)#(?,)
                # session_2
                session_input_2 = tf.concat(
                    [self.item_session_2_embedding, self.cate_session_2_embedding ], 2
                )#(?,20,40)
                mask_2 = self.iterator.session_2_mask#(?, 20)
                real_mask_2 = tf.cast( mask_2, tf.float32)
                session_length_2 = tf.reduce_sum(  mask_2, 1)#(?,)
                # session_3
                session_input_3 = tf.concat(
                    [self.item_session_3_embedding, self.cate_session_3_embedding  ], 2
                )#(?,20,40)
                mask_3 = self.iterator.session_3_mask#(?, 20)
                real_mask_3 = tf.cast( mask_3, tf.float32)
                session_length_3 = tf.reduce_sum( mask_3, 1)#(?,)
                
                session_input_4 = tf.concat(
                    [self.item_session_4_embedding, self.cate_session_4_embedding  ], 2
                )#(?,20,40)
                mask_4 = self.iterator.session_4_mask#(?, 20)
                real_mask_4 = tf.cast( mask_4, tf.float32)
                session_length_4 = tf.reduce_sum( mask_4, 1)#(?,)
             
            
            
                with tf.variable_scope("long_term"):
                    
                    
                   # long_query_emb = tf.reduce_mean(sequence_pre_input, 1) 
                    long_query_emb =    tf.concat([self.user_long_embedding  , self.target_item_embedding], -1)       
                    #long_query_emb =  tf.concat([self.user_long_embedding  ,  tf.reduce_mean(sequence_pre_input, 1) ], -1) 
                    #long_query_emb =  tf.concat([tf.reduce_mean(sequence_pre_input, 1)  , self.target_item_embedding  ], -1) 
                    #long_query_emb =    self.target_item_embedding   
                    with tf.variable_scope("long_term_session"):
                        att_fea_long_0 = self._build_sasrec(session_input_0 +self.position_embedding_long ,real_mask_0 , session_length_0 ,scope='sess_long',num_heads=2,causality=False  )#long-interest公用一个sasrec
                        att_fea_long_1 = self._build_sasrec(session_input_1 +self.position_embedding_long , real_mask_1, session_length_1,scope='sess_long',num_heads=2,causality=False  )
                        att_fea_long_2 = self._build_sasrec( session_input_2 +self.position_embedding_long,real_mask_2, session_length_2,scope='sess_long',num_heads=2,causality=False)
                        att_fea_long_3 = self._build_sasrec(session_input_3 +self.position_embedding_long,real_mask_3, session_length_3,scope='sess_long',num_heads=2,causality=False  )
                        att_fea_long_4 = self._build_sasrec(session_input_4 +self.position_embedding_long,real_mask_4, session_length_4,scope='sess_long',num_heads=2,causality=False  )
                        
                        att_fea_long_0 = tf.reduce_sum(self._attention_fcn(  self.target_item_embedding ,att_fea_long_0,real_mask_0 ),1)
                        att_fea_long_1 = tf.reduce_sum(self._attention_fcn(  self.target_item_embedding ,att_fea_long_1,real_mask_1 ),1)
                        att_fea_long_2 = tf.reduce_sum(self._attention_fcn(  self.target_item_embedding ,att_fea_long_2,real_mask_2 ),1)
                        att_fea_long_3 = tf.reduce_sum(self._attention_fcn(  self.target_item_embedding ,att_fea_long_3,real_mask_3 ),1)
                        att_fea_long_4 = tf.reduce_sum(self._attention_fcn(  self.target_item_embedding ,att_fea_long_4,real_mask_4 ),1)
                        
                    # att_fea_long_0 = tf.reduce_mean(att_fea_long_0, 1)
                    # att_fea_long_1 = tf.reduce_mean(att_fea_long_1, 1)
                    # att_fea_long_2 = tf.reduce_mean(att_fea_long_2, 1)
                    # att_fea_long_3 = tf.reduce_mean(att_fea_long_3, 1)
                    # att_fea_long_4 = tf.reduce_mean(att_fea_long_4, 1)
                     
                     
                    
                    
                    
                    #0是离target最近的session 
                    sess_sasrec_emd= [tf.expand_dims(att_fea_long_4,axis=1), tf.expand_dims(att_fea_long_3,axis=1),tf.expand_dims(att_fea_long_2,axis=1),tf.expand_dims(att_fea_long_1,axis=1),tf.expand_dims(att_fea_long_0,axis=1)]
                    sess_sasrec_emd = concat_func(sess_sasrec_emd, axis=1)#shape=(?, 5, 40)
                     
                     
                    sess_lstm_emd ,_ = dynamic_rnn(
                        tf.nn.rnn_cell.GRUCell(hparams.user_embedding_dim),
                        inputs=  sess_sasrec_emd ,
                        sequence_length= tf.reduce_sum( self.iterator.session_count_mask, 1),
                       # initial_state=self.user_short_embedding,
                        dtype=tf.float32,
                        scope="long_term_rnn",
                    )
                    
                     
            
                    att_fea_long_rnn =self._attention_fcn(long_query_emb, sess_lstm_emd ,self.iterator.session_count_mask)
                    att_fea_long_sasrec =self._attention_fcn(long_query_emb, sess_sasrec_emd,self.iterator.session_count_mask)
                    self.att_fea_long_rnn = tf.reduce_sum(att_fea_long_rnn, 1)
                    self.att_fea_long_sasrec = tf.reduce_sum(att_fea_long_sasrec, 1)
                    self.att_fea_long=tf.concat([self.att_fea_long_rnn, self.att_fea_long_sasrec ],1)

                    
                
                with tf.variable_scope("short_term"):

                    with  tf.variable_scope("short_term_current",reuse=tf.AUTO_REUSE):

                        att_fea_current_pre_rnn = self._build_sasrec(current_session_pre_input  ,real_mask_current_session_pre , self.current_session_pre_length,scope='sess_long',num_heads=2,causality=False  )#long-interest公用一个sasrec

                        # att_fea_current_pre_rnn , short_term_intention = dynamic_rnn(
                        #     tf.nn.rnn_cell.GRUCell(hparams.user_embedding_dim),
                        #     inputs=current_session_pre_input,
                        #     sequence_length= self.current_session_pre_length,
                        #     initial_state=self.user_short_embedding,
                        #     dtype=tf.float32,
                        #     scope="short_term_intention",
                        # )
                        self.att_fea_current_pre_rnn= tf.reduce_sum(self._attention_fcn(self.item_embedding,  att_fea_current_pre_rnn ,real_mask_current_session_pre ),1)
                        att_fea_current_pre_rnn_1 = self._build_sasrec(current_session_pre_1_input  ,real_mask_current_session_pre_1 , self.current_session_pre_1_length,scope='sess_long',num_heads=2,causality=False  )#long-interest公用一个sasrec

                        # att_fea_current_pre_rnn_1 ,_ = dynamic_rnn(
                        #     tf.nn.rnn_cell.GRUCell(hparams.user_embedding_dim),
                        #     inputs=current_session_pre_1_input,
                        #     sequence_length= self.current_session_pre_1_length  ,
                        #     initial_state=self.user_short_embedding,
                        #     dtype=tf.float32,
                        #     scope="short_term_intention",
                             
                        # )
                        att_fea_current_pre_rnn_2 = self._build_sasrec(current_session_pre_2_input   ,real_mask_current_session_pre_2 , self.current_session_pre_2_length,scope='sess_long',num_heads=2,causality=False  )#long-interest公用一个sasrec

                        # att_fea_current_pre_rnn_2, _ = dynamic_rnn(
                        #     tf.nn.rnn_cell.GRUCell(hparams.user_embedding_dim),
                        #     inputs=current_session_pre_2_input,
                        #     sequence_length= self.current_session_pre_2_length  ,
                        #     initial_state=self.user_short_embedding,
                        #     dtype=tf.float32,
                        #     scope="short_term_intention",
                             
                        # )
                     
                        #self.att_fea_current_pre_rnn_1 = tf.reduce_sum(self._attention_fcn(self.item_embedding,att_fea_current_pre_rnn_1,real_mask_current_session_pre_1 ),1)
                        #self.att_fea_current_pre_rnn_2 = tf.reduce_sum(self._attention_fcn(self.item_embedding,att_fea_current_pre_rnn_2,real_mask_current_session_pre_2 ),1)
                        self.att_fea_current_pre_rnn_1 =tf.truediv(tf.reduce_sum(  att_fea_current_pre_rnn_1*tf.expand_dims(real_mask_current_session_pre_1,-1) ,1),tf.expand_dims(tf.cast(self.current_session_pre_1_length, tf.float32),-1)) 
                        self.att_fea_current_pre_rnn_2 =  tf.truediv(tf.reduce_sum(  att_fea_current_pre_rnn_2*tf.expand_dims(real_mask_current_session_pre_2,-1) ,1),tf.expand_dims(tf.cast(self.current_session_pre_2_length , tf.float32),-1))




                    sequence_input_new = tf.concat(
                        [
                           sequence_input ,
                            tf.expand_dims(self.iterator.time_from_first_action[:,20:   ], -1),
                        ],
                        -1,
                    )
                    sequence_input_new = tf.concat(
                        [
                            sequence_input_new,
                            tf.expand_dims(self.iterator.time_to_now[:,20:   ], -1),
                        ],
                        -1,
                    )

                    
                    att_fea_short_rnn , _ = dynamic_rnn(
                        Time4LSTMCell(  hparams.hidden_size  ),
                        inputs= sequence_input_new ,
                        sequence_length=sequence_length ,
                        #initial_state=self.user_short_embedding ,
                        dtype=tf.float32,
                        scope="forward_rnn",
                    )#shape=(?, 50, 40)

                     

                   # short_query_emb =   tf.concat([short_term_intention , self.target_item_embedding], -1)
                    short_query_emb =     self.target_item_embedding    
                    self.att_fea_short_rnn   =tf.reduce_sum(self._attention_fcn(short_query_emb, att_fea_short_rnn     , real_mask_sequence),1)
                    self.att_fea_short  = tf.concat([self.att_fea_short_rnn,self.att_fea_current_pre_rnn],1)
                    

                                    
                
                with tf.name_scope("alpha"):
                    _, final_state = dynamic_rnn(
                                tf.nn.rnn_cell.GRUCell(hparams.hidden_size),
                                inputs=sequence_input,
                                sequence_length= sequence_length,
                                dtype=tf.float32,
                                scope="causal2",
                            )
                    alpha_logit  = self._fcn_net(
                        tf.concat([ final_state,
                                  self.att_fea_long,
                                   self.att_fea_short    ,
                                self.target_item_embedding,tf.expand_dims(self.iterator.time_to_now[:,20:][:, -1], -1)],1), hparams.alpha_fcn_layer_sizes, scope="alpha_fcn"
                    )
                    self.alpha_output  = tf.sigmoid(alpha_logit  )
                    tf.summary.histogram("alpha_all_forward", self.alpha_output  )

                user_LS_embed   =self.att_fea_long* self.alpha_output  + self.att_fea_short   * (1.0 - self.alpha_output  )
                #user_LS_embed   =    self.att_fea_short_rnn
                model_output = tf.concat([user_LS_embed ,self.target_item_embedding], 1)
     
       
                 
        return model_output

     

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

    def train(self, sess, feed_dict,all_step):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        feed_dict[self.is_train_stage] = True
        
        return sess.run(
            [
                self.update,
                self.extra_update_ops,
                self.loss,
                self.forward_data_loss,
                 
               self.contrastive_loss,
                self.regular_loss,
         
              
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def batch_train(self, file_iterator, train_sess, valid_file,valid_num_ngs ,epoch,eval_metric):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        epoch_forward_data_loss = 0
        epoch_contrastive_loss =0
        epoch_regular_loss = 0
       
        for batch_data_input in file_iterator:
            if batch_data_input:
               
                step_result = self.train(train_sess, batch_data_input,self.all_step)
                (_, _, step_loss, step_forward_data_loss, step_contrastive_loss,  step_regular_loss,    summary) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                epoch_forward_data_loss += step_forward_data_loss
                epoch_contrastive_loss +=step_contrastive_loss
                epoch_regular_loss += step_regular_loss
                 
                step += 1
                self.all_step +=1#总体iteration
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, forward_data_loss: {2:.4f} , contrastive_loss: {3:.4f}    ".format(
                            step, step_loss, step_forward_data_loss ,step_contrastive_loss
                        )
                    )
                
                if step % self.hparams.valid_step == 0:
                     
                    valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
                    print(
                        "eval valid at epoch {0},step {1}: {2}".format(
                            epoch,step,
                            ",".join(
                                [
                                    "" + str(key) + ":" + str(value)
                                    for key, value in valid_res.items()
                                ]
                            ),
                        )
                    )
                    print("valid_stop_patience :",self.valid_stop_patience)
                    self.eval_info.append((epoch,step, valid_res))

                    progress = False
                    early_stop = self.hparams.EARLY_STOP
                    if valid_res[eval_metric] > self.best_metric:
                        self.valid_stop_patience=0#累积超过多轮才会break，只要有一次进步都会清零
                        self.best_metric = valid_res[eval_metric]
                        self.best_epoch = epoch
                        progress = True#有长进，可以进行存储
                    else:
                        self.valid_stop_patience +=1
                        if early_stop > 0 and self.valid_stop_patience >= early_stop:
                            print("early stop at epoch {0},step{1}!".format(epoch,step))
                            break

                    if self.hparams.save_model and self.hparams.MODEL_DIR:
                        if not os.path.exists(self.hparams.MODEL_DIR):
                            os.makedirs(self.hparams.MODEL_DIR)
                        if progress:
                            checkpoint_path = self.saver.save(
                                sess=train_sess,
                                save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch)+"_step_" + str(step),
                            )

                 

        return epoch_loss 

    def _add_summaries(self):
       
        tf.summary.scalar("forward_data_loss", self.forward_data_loss)
        
       
  
       
        tf.summary.scalar("contrastive_loss", self.contrastive_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        
        tf.summary.scalar("loss", self.loss)
 
        
        merged = tf.summary.merge_all()
        return merged
    
    def run_weighted_eval_write(self, infile_name, outfile_name,num_ngs,calc_mean_alpha=True ):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            obj: An instance of self.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
         

        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(
                infile_name, batch_num_ngs=0
            ):
                if batch_data_input:
                    if not calc_mean_alpha:
                        step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                    else:
                        step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    
                    
                    users.extend(np.reshape(step_user, -1))
                    preds.extend(np.reshape(step_pred, -1))
                    labels.extend(np.reshape(step_labels, -1))
                    group_preds.extend(np.reshape(step_pred, (-1, group)))
                    group_labels.extend(np.reshape(step_labels, (-1, group)))
                    step_user=np.reshape(step_user, (-1, group)) 
                    step_labels = np.reshape(step_labels,(-1, group)) 
                    step_pred = np.reshape(step_pred, (-1, group)) 
                    step_alpha = np.reshape(step_alpha, (-1, group)) 
                     
                    for i in range(0,step_user.shape[0]):
                        wt.write( '\n'+"\t".join( map(str,step_user[i]))+ '\n'+"\t".join( map(str,step_pred[i]))+ '\n'+"\t".join( map(str,step_labels[i]))+ '\n'+"\t".join( map(str,step_alpha[i]))
                             )
            res = cal_metric(labels, preds, self.hparams.metrics)
            res_pairwise = cal_metric(
                group_labels, group_preds, self.hparams.pairwise_metrics
            )
            res.update(res_pairwise)
            res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
            res.update(res_weighted)
        
                   
        return res
