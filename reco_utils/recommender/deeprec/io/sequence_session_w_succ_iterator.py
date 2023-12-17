 # Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import json
import pickle as pkl
import random
import os
import time

from reco_utils.recommender.deeprec.io.iterator import BaseIterator
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
import pdb 

__all__ = ["SequenceSessionSuccLSIterator" ,"SequenceSessionMaskLSIterator","SequenceSessionCropLSIterator","SequenceSessionSuccExLSIterator","SequenceSessionReorderLSIterator"]

class SequenceSessionSuccLSIterator(BaseIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        self.col_spliter = col_spliter
      

        self.max_seq_length = hparams.max_seq_length
        self.max_session_count =hparams.max_session_count
        self.batch_size = hparams.batch_size
        self.recent_k = hparams.contrastive_recent_k
        self.train_iter_data = dict()

       # self.time_unit = hparams.time_unit

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, 1], name="label")
            self.users = tf.placeholder(tf.int32, [None], name="users")
            self.index = tf.placeholder(tf.float32, [None], name="index")
            self.items = tf.placeholder(tf.int32, [None], name="items")
            self.cates = tf.placeholder(tf.int32, [None], name="cates")
            self.behaviors = tf.placeholder(tf.int32, [None], name="behaviors")
            self.session_count_mask = tf.placeholder(
                tf.int32, [None, self.max_session_count], name="session_count_mask"
            )#[-4,-3,-2,-1 ,-0]#5 这个跟之前的模型不一样了

            self.item_session_0 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_session_0"
            )
            self.item_cate_session_0 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_session_0"
            )
            self.session_0_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="session_0_mask"
            )
            self.item_session_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_session_1"
            )
            self.item_cate_session_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_session_1"
            )
            self.session_1_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="session_1_mask"
            )
            self.item_session_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_session_2"
            )
            self.item_cate_session_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_session_2"
            )
            self.session_2_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="session_2_mask"
            )
            self.item_session_3 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_session_3"
            )
            self.item_cate_session_3 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_session_3"
            )
            self.session_3_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="session_3_mask"
            )
            self.item_session_4 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_session_4"
            )
            self.item_cate_session_4 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_session_4"
            )
            self.session_4_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="session_4_mask"
            )


            #successor
            self.item_sequence = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_sequence"
            )
            self.item_cate_sequence = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_cate_sequence"
            )
            self.sequence_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="sequence_mask"
            )
            self.time_diff = tf.placeholder(
                tf.float32, [None, self.max_seq_length*self.max_session_count], name="time_diff"
            )
            self.time_from_first_action = tf.placeholder(
                tf.float32, [None, self.max_seq_length*self.max_session_count], name="time_from_first_action"
            )
            self.time_to_now = tf.placeholder(
                tf.float32, [None, self.max_seq_length*self.max_session_count], name="time_to_now"
            )


            self.item_sequence_pre = tf.placeholder(
                tf.int32, [None,self.recent_k], name="item_sequence_pre"
            )
            self.item_cate_sequence_pre = tf.placeholder(
                tf.int32, [None, self.recent_k], name="item_cate_sequence_pre"
            )

            self.item_current_session_pre = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre"
            )
            self.item_cate_current_session_pre = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre"
            )
            self.current_session_pre_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask"
            )

            self.item_current_session_succ = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_succ"
            )
            self.item_cate_current_session_succ = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_succ"
            )
            self.current_session_succ_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_succ_mask"
            )

            

    def parse_file(self, input_file):
        """Parse the file to a list ready to be used for downstream tasks
        
        Args:
            input_file: One of train, valid or test file which has never been parsed.
        
        Returns: 
            list: A list with parsing result
        """
        with open(input_file, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def parser_one_line(self, line):
        """Parse one string line into feature values.
            a line was saved as the following format:
            label \t index \t user_hash \t item_hash \t item_cate \t behavior_type\t sess_state \t 
            item_session_succ  \t item_cate_session_succ \t behavior_type_session_succ \t sess_state_session_succ \t
            item_session_0  \t item_cate_session_0 \t behavior_type_session_0 \t sess_state_session_0 \t
            item_session_1  \t item_cate_session_1 \t behavior_type_session_1 \t sess_state_session_1 \t
            item_session_2  \t item_cate_session_2 \t behavior_type_session_2 \t sess_state_session_2 \t
            item_session_3  \t item_cate_session_3 \t behavior_type_session_3 \t sess_state_session_3 \t
            item_session_4  \t item_cate_session_4 \t behavior_type_session_4 \t sess_state_session_4 \t
            valid_sess \n[1,2,3,4]

        Args:
            line (str): a string indicating one instance

        Returns:
            tuple/list: Parsed results including label, user_id, target_item_id, target_category, item_history, cate_history(, timeinterval_history,
            timelast_history, timenow_history, mid_mask, seq_len, learning_rate)

        """

        words = line.strip().split(self.col_spliter)
        label = float(words[0])
       # index = float(words[1])#用于记录loss的变化
        user_id =  int(words[2])
        item_id = int(words[3])
        item_cate = int(words[4])
        item_ts = float(words[5])
        item_behavior = int(words[6])
       

        item_sequence_words = words[7].strip('[').strip(']').split(", ")
        cate_sequence_words = words[8].strip('[').strip(']').split(", ")
        item_sequence    = [int(i) for i in item_sequence_words]
        cate_sequence   = [int(i) for i in cate_sequence_words ]


        time_sequence_words = words[9].strip('[').strip(']').split(", ")
        time_sequence = [float(i) for i in time_sequence_words]


        time_range = 3600 * 24 / 1000

        time_diff = []
        for i in range(len(time_sequence) - 1):
            diff = (
                time_sequence[i + 1] - time_sequence[i]
            ) / time_range
            diff = max(diff, 0.5)
            time_diff.append(diff)
        last_diff = (item_ts - time_sequence[-1]) / time_range
        last_diff = max(last_diff, 0.5)
        time_diff.append(last_diff)
        time_diff = np.log(time_diff)

        time_from_first_action = []
        first_time = time_sequence[0]
        time_from_first_action = [
            (t - first_time) / time_range for t in time_sequence[1:]
        ]
        time_from_first_action = [max(t, 0.5) for t in time_from_first_action]
        last_diff = (item_ts - first_time) / time_range
        last_diff = max(last_diff, 0.5)
        time_from_first_action.append(last_diff)
        time_from_first_action = np.log(time_from_first_action)

        time_to_now = []
        time_to_now = [(item_ts - t) / time_range for t in time_sequence]
        time_to_now = [max(t, 0.5) for t in time_to_now]
        time_to_now = np.log(time_to_now)


        item_session_4_words = words[10].strip('[').strip(']').split(", ")
        cate_session_4_words = words[11].strip('[').strip(']').split(", ")
        item_session_4   = [int(i) for i in item_session_4_words]
        cate_session_4  = [int(i) for i in cate_session_4_words ]
        item_session_3_words = words[12].strip('[').strip(']').split(", ")
        cate_session_3_words = words[13].strip('[').strip(']').split(", ")
        item_session_3   = [int(i) for i in item_session_3_words]
        cate_session_3  = [int(i) for i in cate_session_3_words ]
        item_session_2_words = words[14].strip('[').strip(']').split(", ")
        cate_session_2_words = words[15].strip('[').strip(']').split(", ")
        item_session_2   = [int(i) for i in item_session_2_words]
        cate_session_2  = [int(i) for i in cate_session_2_words ]
        item_session_1_words = words[16].strip('[').strip(']').split(", ")
        cate_session_1_words = words[17].strip('[').strip(']').split(", ")
        item_session_1   = [int(i) for i in item_session_1_words]
        cate_session_1  = [int(i) for i in cate_session_1_words ]
        item_session_0_words = words[18].strip('[').strip(']').split(", ")
        cate_session_0_words = words[19].strip('[').strip(']').split(", ")
        item_session_0   = [int(i) for i in item_session_0_words]
        cate_session_0  = [int(i) for i in cate_session_0_words ]
        pdb.set_trace()
        valid_sess_words = words[20].strip('[').strip(']').split(", ")
      
        valid_sess =  [int(i) for i in valid_sess_words ]#顺序是[-4,-3,-2,-1 ,0]
         
        item_sequence_pre    = item_sequence[-self.recent_k:]
        cate_sequence_pre   = cate_sequence[-self.recent_k:]

        item_current_session_pre_words = words[21].strip('[').strip(']').split(", ")
        item_current_session_pre  = [int(i) for i in item_current_session_pre_words]

        cate_current_session_pre_words = words[22].strip('[').strip(']').split(", ")
        cate_current_session_pre  = [int(i) for i in cate_current_session_pre_words]

        item_current_session_succ_words = words[23].strip('[').strip(']').split(", ")
        item_current_session_succ  = [int(i) for i in item_current_session_succ_words]

        cate_current_session_succ_words = words[24].strip('[').strip(']').split(", ")
        cate_current_session_succ  = [int(i) for i in cate_current_session_succ_words]
          

        
       
 

        return (
            label,
        
            user_id,
            item_id,
            item_cate,
            item_ts,
            item_behavior,

            item_sequence  ,
            cate_sequence,
            time_diff,
            time_from_first_action,
            time_to_now,




            item_session_0 ,
            cate_session_0 ,
            item_session_1 ,
            cate_session_1 ,
            item_session_2 ,
            cate_session_2 ,

            item_session_3 ,
            cate_session_3 ,
            item_session_4 ,
            cate_session_4 ,
            valid_sess,
             item_sequence_pre  ,
            cate_sequence_pre,

            item_current_session_pre,
            cate_current_session_pre,
            item_current_session_succ ,
            cate_current_session_succ , 
         
        )

     

     

    def load_data_from_file(self, infile, batch_num_ngs=0, min_seq_length=1):
        """Read and parse data from a file.
        
        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_num_ngs (int): The number of negative sampling here in batch. 
                0 represents that there is no need to do negative sampling here.
            min_seq_length (int): The minimum number of a sequence length. 
                Sequences with length lower than min_seq_length will be ignored.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """
       
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_ts_list =[]
        item_behavior_list =[]
         
        item_sequence_batch = []
        item_cate_sequence_batch = []
        time_diff_batch = []
        time_from_first_action_batch = []
        time_to_now_batch = []

        item_session_0_batch = []
        item_cate_session_0_batch = []
        item_session_1_batch = []
        item_cate_session_1_batch = []
        item_session_2_batch = []
        item_cate_session_2_batch = []
        item_session_3_batch = []
        item_cate_session_3_batch = []
        item_session_4_batch = []
        item_cate_session_4_batch = []
        valid_sess_batch=[]
        item_sequence_pre_batch =[]
        item_cate_sequence_pre_batch =[]
        item_current_session_pre_batch =[]
        item_cate_current_session_pre_batch =[]
        item_current_session_succ_batch =[]
        item_cate_current_session_succ_batch =[]


        cnt = 0

        if infile not in self.train_iter_data :
            lines = self.parse_file(infile)
            self.train_iter_data[infile] = lines
        else:
            lines = self.train_iter_data[infile]

        if batch_num_ngs > 0:
            random.shuffle(lines)

        for line in lines:
            if not line:
                continue

            (
                label,
         
                user_id,
                item_id,
                item_cate,
                item_ts,
                item_behavior,
                
                item_sequence,
                item_cate_sequence,
               
                time_diff,
                time_from_first_action,
                time_to_now,

                item_session_0,
                item_cate_session_0,
                item_session_1,
                item_cate_session_1,
                item_session_2,
                item_cate_session_2,
                item_session_3,
                item_cate_session_3,
                item_session_4,
                item_cate_session_4,
                valid_sess,
                item_sequence_pre,
                item_cate_sequence_pre,

                item_current_session_pre,
                item_cate_current_session_pre,
                item_current_session_succ,
                item_cate_current_session_succ,


                 
            ) = line
           

            label_list.append(label)
            
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_ts_list.append(item_ts)
            item_behavior_list.append(item_behavior)
            
            item_sequence_batch.append(item_sequence)
            item_cate_sequence_batch.append(item_cate_sequence)
            time_diff_batch.append(time_diff)
            time_from_first_action_batch.append(time_from_first_action)
            time_to_now_batch.append(time_to_now)

            item_session_0_batch.append(item_session_0)
            item_cate_session_0_batch.append(item_cate_session_0)
            item_session_1_batch.append(item_session_1)
            item_cate_session_1_batch.append(item_cate_session_1)
            item_session_2_batch.append(item_session_2)
            item_cate_session_2_batch.append(item_cate_session_2)
            item_session_3_batch.append(item_session_3)
            item_cate_session_3_batch.append(item_cate_session_3)
            item_session_4_batch.append(item_session_4)
            item_cate_session_4_batch.append(item_cate_session_4)

            valid_sess_batch.append(valid_sess)
            item_sequence_pre_batch.append(item_sequence_pre)
            item_cate_sequence_pre_batch.append(item_cate_sequence_pre)
            item_current_session_pre_batch.append(item_current_session_pre)
            item_cate_current_session_pre_batch.append( item_cate_current_session_pre)
            item_current_session_succ_batch.append(item_current_session_succ)
            item_cate_current_session_succ_batch.append(item_cate_current_session_succ)
           
            cnt += 1
            if cnt == self.batch_size:
                res = self._convert_data(
                    label_list,
             
                    user_list,
                    item_list,
                    item_cate_list,
                    item_ts_list,
                    item_behavior_list,
                   
                    item_sequence_batch,
                    item_cate_sequence_batch,
                    time_diff_batch,
                    time_from_first_action_batch,
                    time_to_now_batch,

                    item_session_0_batch ,
                    item_cate_session_0_batch ,
                    item_session_1_batch ,
                    item_cate_session_1_batch ,
                    item_session_2_batch ,
                    item_cate_session_2_batch, 
                    item_session_3_batch ,
                    item_cate_session_3_batch ,
                    item_session_4_batch ,
                    item_cate_session_4_batch ,

                    valid_sess_batch ,
                    item_sequence_pre_batch ,
                    item_cate_sequence_pre_batch ,
                     item_current_session_pre_batch,
                    item_cate_current_session_pre_batch,
                    item_current_session_succ_batch,
                    item_cate_current_session_succ_batch ,
                
                    batch_num_ngs,
                )
                batch_input = self.gen_feed_dict(res)
                yield batch_input if batch_input else None
                label_list = []
                
                user_list = []
                item_list = []
                item_cate_list = []
                item_ts_list =[]
                item_behavior_list =[]
                
                item_sequence_batch = []
                item_cate_sequence_batch = []
                time_diff_batch = []
                time_from_first_action_batch = []
                time_to_now_batch = []

                item_session_0_batch= []
                item_cate_session_0_batch= []
                item_session_1_batch= []
                item_cate_session_1_batch= []
                item_session_2_batch= []
                item_cate_session_2_batch= []
                item_session_3_batch= []
                item_cate_session_3_batch= []
                item_session_4_batch= []
                item_cate_session_4_batch= []
                valid_sess_batch= []
                item_sequence_pre_batch =[]
                item_cate_sequence_pre_batch =[]
                item_current_session_pre_batch =[]
                item_cate_current_session_pre_batch =[]
                item_current_session_succ_batch =[]
                item_cate_current_session_succ_batch =[]
                cnt = 0
        if cnt > 0:
            res = self._convert_data(
                label_list,
                 
                    user_list,
                    item_list,
                    item_cate_list,
                     item_ts_list,
                    item_behavior_list,
                   
                    item_sequence_batch,
                    item_cate_sequence_batch,
                    time_diff_batch,
                    time_from_first_action_batch,
                    time_to_now_batch,
                    item_session_0_batch ,
                    item_cate_session_0_batch ,
                    item_session_1_batch ,
                    item_cate_session_1_batch ,
                    item_session_2_batch ,
                    item_cate_session_2_batch, 
                    item_session_3_batch ,
                    item_cate_session_3_batch ,
                    item_session_4_batch ,
                    item_cate_session_4_batch ,

                    valid_sess_batch ,
                    item_sequence_pre_batch ,
                    item_cate_sequence_pre_batch ,

                    item_current_session_pre_batch,
                    item_cate_current_session_pre_batch,
                    item_current_session_succ_batch,
                    item_cate_current_session_succ_batch ,

                batch_num_ngs,
            )
            batch_input = self.gen_feed_dict(res)
            yield batch_input if batch_input else None

    def _convert_data(
        self,
       label_list,
       
        user_list,
        item_list,
        item_cate_list,
        item_ts_list,
        item_behavior_list,
        
        item_sequence_batch,
        item_cate_sequence_batch,
        time_diff_batch,
        time_from_first_action_batch,
        time_to_now_batch,
        item_session_0_batch ,
        item_cate_session_0_batch ,
        item_session_1_batch ,
        item_cate_session_1_batch ,
        item_session_2_batch ,
        item_cate_session_2_batch, 
        item_session_3_batch ,
        item_cate_session_3_batch ,
        item_session_4_batch ,
        item_cate_session_4_batch ,

        valid_sess_batch ,
        item_sequence_pre_batch ,
        item_cate_sequence_pre_batch ,
        item_current_session_pre_batch,
        item_cate_current_session_pre_batch,
        item_current_session_succ_batch,
        item_cate_current_session_succ_batch ,
        batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            time_list (list): a list of current timestamp.
            time_diff_list (list): a list of timestamp between each sequential opertions.
            time_from_first_action_list (list): a list of timestamp from the first opertion.
            time_to_now_list (list): a list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return
            # 标量 变量
            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            #标量 不变量
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            
            behavior_list_all = np.asarray(
                [[ behavior] * (batch_num_ngs + 1) for  behavior in  item_behavior_list], dtype=np.int32
            ).flatten()
        

            # history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]#标量 不变量，后面应该要扩展
            max_seq_length  = self.max_seq_length
            max_session_count= self.max_session_count#4
            # 向量 list 不变量 先扩展位置，填充0

            item_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 



            time_diff_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_to_now_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")



            item_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            
            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k)
            ).astype("int32") 

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            item_current_session_succ_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_succ_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 


            #mask也是list
            mask_all = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_session_count )
            ).astype("float32")
            mask_sequence = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            mask_0 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_3 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_4 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            mask_pre =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            mask_succ =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            #先处理历史序列
            
            for i in range(instance_cnt):
                #已经做过截断了，这里直接用
                    
                #sequence 
                sequence_length = len(item_sequence_batch[i])
                sequence_pre_length = len(item_sequence_pre_batch[i])
                for index in range(batch_num_ngs + 1):
                    item_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_batch[i] , dtype=np.int32)
                    item_cate_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_batch[i] , dtype=np.int32
                    )
                    mask_sequence[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    time_diff_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_diff_batch[i] , dtype=np.float32)
                    time_from_first_action_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        time_from_first_action_batch[i] , dtype=np.float32
                    )
                    time_to_now_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_to_now_batch[i] , dtype=np.float32)


                    item_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length 
                    ] = np.asarray(item_sequence_pre_batch[i] , dtype=np.int32)
                    item_cate_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length
                    ] = np.asarray(
                        item_cate_sequence_pre_batch[i] , dtype=np.int32
                    )
                #current_session
                current_session_pre_length = len(item_current_session_pre_batch[i])
                current_session_succ_length = len(item_current_session_succ_batch[i])
                for index in range(batch_num_ngs + 1):
                    item_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length ] = np.asarray(item_current_session_pre_batch[i] , dtype=np.int32)
                    item_cate_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = np.asarray(item_cate_current_session_pre_batch[i] , dtype=np.int32)
                    mask_pre[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = 1.0
                    
                    item_current_session_succ_batch_all[i * (batch_num_ngs + 1) + index, :current_session_succ_length ] = np.asarray(item_current_session_succ_batch[i] , dtype=np.int32)
                    item_cate_current_session_succ_batch_all[i * (batch_num_ngs + 1) + index, :current_session_succ_length] = np.asarray(item_cate_current_session_succ_batch[i] , dtype=np.int32)
                    mask_succ[i * (batch_num_ngs + 1) + index, :current_session_succ_length] = 1.0


                #long-term session
                #5个长期序列， 不一定存在
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count>0:
                    for index in range(batch_num_ngs + 1):
                        mask_all[i * (batch_num_ngs + 1) + index,  :sess_count ] = 1.0 
                session_4_length =valid_sess_batch[i][0]
                if session_4_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length ] = np.asarray(item_session_4_batch[i] , dtype=np.int32)
                        item_cate_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length] = np.asarray(item_cate_session_4_batch[i] , dtype=np.int32)
                        mask_4[i * (batch_num_ngs + 1) + index, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length =valid_sess_batch[i][1]
                if session_3_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length ] = np.asarray(item_session_3_batch[i] , dtype=np.int32)
                        item_cate_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length] = np.asarray(item_cate_session_3_batch[i] , dtype=np.int32)
                        mask_3[i * (batch_num_ngs + 1) + index, :session_3_length] = 1.0
                else:
                    continue
                session_2_length =valid_sess_batch[i][2]
                if session_2_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length ] = np.asarray(item_session_2_batch[i] , dtype=np.int32)
                        item_cate_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length] = np.asarray(item_cate_session_2_batch[i] , dtype=np.int32)
                        mask_2[i * (batch_num_ngs + 1) + index, :session_2_length] = 1.0
                else:
                    continue
                session_1_length =valid_sess_batch[i][3]
                if session_1_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length ] = np.asarray(item_session_1_batch[i] , dtype=np.int32)
                        item_cate_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length] = np.asarray(item_cate_session_1_batch[i] , dtype=np.int32)
                        mask_1[i * (batch_num_ngs + 1) + index, :session_1_length] = 1.0
                else:
                    continue
                session_0_length =valid_sess_batch[i][4]
                if session_0_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length ] = np.asarray(item_session_0_batch[i] , dtype=np.int32)
                        item_cate_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length] = np.asarray(item_cate_session_0_batch[i] , dtype=np.int32)
                        mask_0[i * (batch_num_ngs + 1) + index, :session_0_length] = 1.0
                


            #处理负采样
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
           
             
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
             
            res["users"] = np.asarray(user_list_all, dtype=np.int32)
            res["behaviors"] = np.asarray(behavior_list_all, dtype=np.int32)
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            

            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all

            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0
            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1
            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4

            res["session_count_mask"] = mask_all

            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre
            res["item_current_session_succ"] = item_current_session_succ_batch_all
            res["item_cate_current_session_succ"] = item_cate_current_session_succ_batch_all
            res["current_session_succ_mask"] = mask_succ


        else:
            #测试和验证
            instance_cnt = len(label_list)
             
            max_seq_length  = self.max_seq_length
            max_session_count =self.max_session_count


            item_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")

            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt  , self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt  ,  self.recent_k )
            ).astype("int32")

            time_diff_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt, max_seq_length* max_session_count  )
            ).astype("float32")
            time_to_now_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            mask_sequence = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")    


            item_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_0 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_1= np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_2 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_3 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_4 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            mask_all = np.zeros((instance_cnt, max_session_count )).astype("float32")

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


            item_current_session_succ_batch_all = np.zeros(
                (instance_cnt , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_succ_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 
            

            mask_succ =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")

   
            for i in range(instance_cnt):
                #sequence
                sequence_length = len(item_sequence_batch[i])
                item_sequence_batch_all[i, :sequence_length] = item_sequence_batch[i] 
                item_cate_sequence_batch_all[i, :sequence_length] = item_cate_sequence_batch[i] 
                sequence_pre_length = len(item_sequence_pre_batch[i])
                item_sequence_pre_batch_all[i, :sequence_pre_length] = item_sequence_pre_batch[i] 
                item_cate_sequence_pre_batch_all[i, :sequence_pre_length] = item_cate_sequence_pre_batch[i] 

                mask_sequence[i, :sequence_length] = 1.0
                time_diff_batch_all[i, :sequence_length] = time_diff_batch[i] 
                time_from_first_action_batch_all[i, :sequence_length] = time_from_first_action_batch[i] 
                time_to_now_batch_all[i, :sequence_length] = time_to_now_batch[i] 
                #succ 无需处理，保持为0即可
                current_session_pre_length = len(item_current_session_pre_batch[i])
                item_current_session_pre_batch_all[i  , :current_session_pre_length ] =  item_current_session_pre_batch[i]  
                item_cate_current_session_pre_batch_all[i  , :current_session_pre_length] =  item_cate_current_session_pre_batch[i]  
                mask_pre[i , :current_session_pre_length] = 1.0
                

                #session
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count >0 :  
                    mask_all[i, :sess_count ] = 1.0

                session_4_length = valid_sess_batch[i][0] 
                if session_4_length >0:
                  
                    item_session_4_batch_all[i, :session_4_length] = item_session_4_batch[i] 
                    item_cate_session_4_batch_all[i, :session_4_length] = item_cate_session_4_batch[i] 
                    mask_4[i, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length = valid_sess_batch[i][1] 
                if session_3_length >0:
                    
                    item_session_3_batch_all[i, :session_3_length] = item_session_3_batch[i] 
                    item_cate_session_3_batch_all[i, :session_3_length] = item_cate_session_3_batch[i] 
                    mask_3[i, :session_3_length] = 1.0
                else:
                    continue
                
                session_2_length = valid_sess_batch[i][2] 
                if session_2_length >0:
                     
                    item_session_2_batch_all[i, :session_2_length] = item_session_2_batch[i] 
                    item_cate_session_2_batch_all[i, :session_2_length] = item_cate_session_2_batch[i] 
                    mask_2[i, :session_2_length] = 1.0
                else:
                    continue
                session_1_length = valid_sess_batch[i][3] 
                if session_1_length >0:
                     
                    item_session_1_batch_all[i, :session_1_length] = item_session_1_batch[i] 
                    item_cate_session_1_batch_all[i, :session_1_length] = item_cate_session_1_batch[i] 
                    mask_1[i, :session_1_length] = 1.0
                else:
                    continue


                session_0_length = valid_sess_batch[i][4] 
                if session_0_length >0:
            
                    item_session_0_batch_all[i, :session_0_length] = item_session_0_batch[i] 
                    item_cate_session_0_batch_all[i, :session_0_length] = item_cate_session_0_batch[i] 
                    mask_0[i, :session_0_length] = 1.0
                    
                
     
            
            res = {}
           
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            
            res["users"] = np.asarray(user_list, dtype=np.int32)
            res["behaviors"] = np.asarray(item_behavior_list, dtype=np.int32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all
            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0

            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1

            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4
            
            

            res["session_count_mask"] = mask_all
            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre
            res["item_current_session_succ"] = item_current_session_succ_batch_all
            res["item_cate_current_session_succ"] = item_cate_current_session_succ_batch_all
            res["current_session_succ_mask"] = mask_succ


        return res

        
            

    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            #self.index: data_dict["index"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.behaviors:data_dict["behaviors"],
            self.item_sequence: data_dict["item_sequence"],
            self.item_cate_sequence: data_dict["item_cate_sequence"],
            self.item_sequence_pre: data_dict["item_sequence_pre"],
            self.item_cate_sequence_pre: data_dict["item_cate_sequence_pre"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"],
            self.sequence_mask : data_dict["sequence_mask"],
            self.item_session_0: data_dict["item_session_0"],
            self.item_cate_session_0: data_dict["item_cate_session_0"],
            self.session_0_mask : data_dict["session_0_mask"],
            self.item_session_1: data_dict["item_session_1"],
            self.item_cate_session_1: data_dict["item_cate_session_1"],
            self.session_1_mask : data_dict["session_1_mask"],
            self.item_session_2: data_dict["item_session_2"],
            self.item_cate_session_2: data_dict["item_cate_session_2"],
            self.session_2_mask : data_dict["session_2_mask"],
            self.item_session_3: data_dict["item_session_3"],
            self.item_cate_session_3: data_dict["item_cate_session_3"],
            self.session_3_mask : data_dict["session_3_mask"],
            self.item_session_4: data_dict["item_session_4"],
            self.item_cate_session_4: data_dict["item_cate_session_4"],
            self.session_4_mask : data_dict["session_4_mask"],

            self.session_count_mask:data_dict["session_count_mask"],

            self.item_current_session_pre: data_dict["item_current_session_pre"],
            self.item_cate_current_session_pre: data_dict["item_cate_current_session_pre"],
            self.current_session_pre_mask : data_dict["current_session_pre_mask"],

            self.item_current_session_succ: data_dict["item_current_session_succ"],
            self.item_cate_current_session_succ: data_dict["item_cate_current_session_succ"],
            self.current_session_succ_mask : data_dict["current_session_succ_mask"],
           
        }
        return feed_dict


class SequenceSessionMaskLSIterator(SequenceSessionSuccLSIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        super(SequenceSessionMaskLSIterator, self).__init__(hparams, graph, col_spliter)
        self.augment_rate =hparams.augment_rate
        with self.graph.as_default():
            self.item_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_1"
            )
            self.item_cate_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_1"
            )
            self.current_session_pre_mask_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_1"
            )

            self.item_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_2"
            )
            self.item_cate_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_2"
            )
            self.current_session_pre_mask_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_2"
            )

    
    def _convert_data(
        self,
       label_list,
       
        user_list,
        item_list,
        item_cate_list,
        item_ts_list,
        item_behavior_list,
        
        item_sequence_batch,
        item_cate_sequence_batch,
        time_diff_batch,
        time_from_first_action_batch,
        time_to_now_batch,
        item_session_0_batch ,
        item_cate_session_0_batch ,
        item_session_1_batch ,
        item_cate_session_1_batch ,
        item_session_2_batch ,
        item_cate_session_2_batch, 
        item_session_3_batch ,
        item_cate_session_3_batch ,
        item_session_4_batch ,
        item_cate_session_4_batch ,

        valid_sess_batch ,
        item_sequence_pre_batch ,
        item_cate_sequence_pre_batch ,
        item_current_session_pre_batch,
        item_cate_current_session_pre_batch,
        item_current_session_succ_batch,
        item_cate_current_session_succ_batch ,
        batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            time_list (list): a list of current timestamp.
            time_diff_list (list): a list of timestamp between each sequential opertions.
            time_from_first_action_list (list): a list of timestamp from the first opertion.
            time_to_now_list (list): a list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return
            # 标量 变量
            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            #标量 不变量
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            
            behavior_list_all = np.asarray(
                [[ behavior] * (batch_num_ngs + 1) for  behavior in  item_behavior_list], dtype=np.int32
            ).flatten()
        

            # history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]#标量 不变量，后面应该要扩展
            max_seq_length  = self.max_seq_length
            max_session_count= self.max_session_count#4
            # 向量 list 不变量 先扩展位置，填充0

            item_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 



            time_diff_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_to_now_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")



            item_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            
            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k)
            ).astype("int32") 

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            

            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 



            #mask也是list
            mask_all = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_session_count )
            ).astype("float32")
            mask_sequence = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            mask_0 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_3 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_4 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            mask_pre =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            

            mask_pre_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_pre_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            #先处理历史序列
            
            for i in range(instance_cnt):
                #已经做过截断了，这里直接用
                    
                #sequence 
                sequence_length = len(item_sequence_batch[i])
                sequence_pre_length = len(item_sequence_pre_batch[i])
                for index in range(batch_num_ngs + 1):
                    item_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_batch[i] , dtype=np.int32)
                    item_cate_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_batch[i] , dtype=np.int32
                    )
                    mask_sequence[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    time_diff_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_diff_batch[i] , dtype=np.float32)
                    time_from_first_action_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        time_from_first_action_batch[i] , dtype=np.float32
                    )
                    time_to_now_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_to_now_batch[i] , dtype=np.float32)


                    item_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length 
                    ] = np.asarray(item_sequence_pre_batch[i] , dtype=np.int32)
                    item_cate_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length
                    ] = np.asarray(
                        item_cate_sequence_pre_batch[i] , dtype=np.int32
                    )
                #current_session

                current_session_pre_length = len(item_current_session_pre_batch[i])
                #data AUgmentation
                
                remove_index_1 = random.sample(list(range(current_session_pre_length)),k=int(np.ceil(self.augment_rate*current_session_pre_length)))
                remove_index_2 = random.sample(list(range(current_session_pre_length)),k=int(np.ceil(self.augment_rate*current_session_pre_length)))


                item_current_session_pre_1 =  [ item_current_session_pre_batch[i][index] if index not in remove_index_1 else 0  for index in range(current_session_pre_length)   ] 
                item_cate_current_session_pre_1 =  [item_cate_current_session_pre_batch[i][index]  if index not in remove_index_1 else 0 for index in range(current_session_pre_length)]

                item_current_session_pre_2=  [item_current_session_pre_batch[i][index] if index not in remove_index_2 else 0 for index in range(current_session_pre_length) ]
                item_cate_current_session_pre_2 =  [item_cate_current_session_pre_batch[i][index] if index not in remove_index_2 else 0 for index in range(current_session_pre_length)  ]

                # current_session_pre_1_length = len(item_current_session_pre_1 )
                # current_session_pre_2_length = len(item_current_session_pre_2)

                current_session_pre_1_length = current_session_pre_length
                current_session_pre_2_length = current_session_pre_length
                
                
                
                for index in range(batch_num_ngs + 1):
                    item_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length ] = np.asarray(item_current_session_pre_batch[i] , dtype=np.int32)
                    item_cate_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = np.asarray(item_cate_current_session_pre_batch[i] , dtype=np.int32)
                    mask_pre[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = 1.0

                    item_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length ] = np.asarray(item_current_session_pre_1 , dtype=np.int32)
                    item_cate_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = np.asarray(item_cate_current_session_pre_1 , dtype=np.int32)
                    mask_pre_1[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = 1.0

                    item_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length ] = np.asarray(item_current_session_pre_2 , dtype=np.int32)
                    item_cate_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = np.asarray(item_cate_current_session_pre_2 , dtype=np.int32)
                    mask_pre_2[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = 1.0

                    
                     

                #long-term session
                #5个长期序列， 不一定存在
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count>0:
                    for index in range(batch_num_ngs + 1):
                        mask_all[i * (batch_num_ngs + 1) + index,  :sess_count ] = 1.0 
                session_4_length =valid_sess_batch[i][0]
                if session_4_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length ] = np.asarray(item_session_4_batch[i] , dtype=np.int32)
                        item_cate_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length] = np.asarray(item_cate_session_4_batch[i] , dtype=np.int32)
                        mask_4[i * (batch_num_ngs + 1) + index, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length =valid_sess_batch[i][1]
                if session_3_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length ] = np.asarray(item_session_3_batch[i] , dtype=np.int32)
                        item_cate_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length] = np.asarray(item_cate_session_3_batch[i] , dtype=np.int32)
                        mask_3[i * (batch_num_ngs + 1) + index, :session_3_length] = 1.0
                else:
                    continue
                session_2_length =valid_sess_batch[i][2]
                if session_2_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length ] = np.asarray(item_session_2_batch[i] , dtype=np.int32)
                        item_cate_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length] = np.asarray(item_cate_session_2_batch[i] , dtype=np.int32)
                        mask_2[i * (batch_num_ngs + 1) + index, :session_2_length] = 1.0
                else:
                    continue
                session_1_length =valid_sess_batch[i][3]
                if session_1_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length ] = np.asarray(item_session_1_batch[i] , dtype=np.int32)
                        item_cate_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length] = np.asarray(item_cate_session_1_batch[i] , dtype=np.int32)
                        mask_1[i * (batch_num_ngs + 1) + index, :session_1_length] = 1.0
                else:
                    continue
                session_0_length =valid_sess_batch[i][4]
                if session_0_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length ] = np.asarray(item_session_0_batch[i] , dtype=np.int32)
                        item_cate_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length] = np.asarray(item_cate_session_0_batch[i] , dtype=np.int32)
                        mask_0[i * (batch_num_ngs + 1) + index, :session_0_length] = 1.0
                


            #处理负采样
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
           
            
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
             
            res["users"] = np.asarray(user_list_all, dtype=np.int32)
            res["behaviors"] = np.asarray(behavior_list_all, dtype=np.int32)
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            

            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all

            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0
            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1
            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4

            res["session_count_mask"] = mask_all

            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2


             


        else:
            #测试和验证
            instance_cnt = len(label_list)
             
            max_seq_length  = self.max_seq_length
            max_session_count =self.max_session_count


            item_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")

            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt  , self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt  ,  self.recent_k )
            ).astype("int32")

            time_diff_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt, max_seq_length* max_session_count  )
            ).astype("float32")
            time_to_now_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            mask_sequence = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")    


            item_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_0 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_1= np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_2 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_3 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_4 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            mask_all = np.zeros((instance_cnt, max_session_count )).astype("float32")

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_1 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_2 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


             

           

   
            for i in range(instance_cnt):
                #sequence
                sequence_length = len(item_sequence_batch[i])
                item_sequence_batch_all[i, :sequence_length] = item_sequence_batch[i] 
                item_cate_sequence_batch_all[i, :sequence_length] = item_cate_sequence_batch[i] 
                sequence_pre_length = len(item_sequence_pre_batch[i])
                item_sequence_pre_batch_all[i, :sequence_pre_length] = item_sequence_pre_batch[i] 
                item_cate_sequence_pre_batch_all[i, :sequence_pre_length] = item_cate_sequence_pre_batch[i] 

                mask_sequence[i, :sequence_length] = 1.0
                time_diff_batch_all[i, :sequence_length] = time_diff_batch[i] 
                time_from_first_action_batch_all[i, :sequence_length] = time_from_first_action_batch[i] 
                time_to_now_batch_all[i, :sequence_length] = time_to_now_batch[i] 
                #succ 无需处理，保持为0即可
                current_session_pre_length = len(item_current_session_pre_batch[i])
                item_current_session_pre_batch_all[i  , :current_session_pre_length ] =  item_current_session_pre_batch[i]  
                item_cate_current_session_pre_batch_all[i  , :current_session_pre_length] =  item_cate_current_session_pre_batch[i]  
                mask_pre[i , :current_session_pre_length] = 1.0
                

                #session
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count >0 :  
                    mask_all[i, :sess_count ] = 1.0

                session_4_length = valid_sess_batch[i][0] 
                if session_4_length >0:
                  
                    item_session_4_batch_all[i, :session_4_length] = item_session_4_batch[i] 
                    item_cate_session_4_batch_all[i, :session_4_length] = item_cate_session_4_batch[i] 
                    mask_4[i, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length = valid_sess_batch[i][1] 
                if session_3_length >0:
                    
                    item_session_3_batch_all[i, :session_3_length] = item_session_3_batch[i] 
                    item_cate_session_3_batch_all[i, :session_3_length] = item_cate_session_3_batch[i] 
                    mask_3[i, :session_3_length] = 1.0
                else:
                    continue
                
                session_2_length = valid_sess_batch[i][2] 
                if session_2_length >0:
                     
                    item_session_2_batch_all[i, :session_2_length] = item_session_2_batch[i] 
                    item_cate_session_2_batch_all[i, :session_2_length] = item_cate_session_2_batch[i] 
                    mask_2[i, :session_2_length] = 1.0
                else:
                    continue
                session_1_length = valid_sess_batch[i][3] 
                if session_1_length >0:
                     
                    item_session_1_batch_all[i, :session_1_length] = item_session_1_batch[i] 
                    item_cate_session_1_batch_all[i, :session_1_length] = item_cate_session_1_batch[i] 
                    mask_1[i, :session_1_length] = 1.0
                else:
                    continue


                session_0_length = valid_sess_batch[i][4] 
                if session_0_length >0:
            
                    item_session_0_batch_all[i, :session_0_length] = item_session_0_batch[i] 
                    item_cate_session_0_batch_all[i, :session_0_length] = item_cate_session_0_batch[i] 
                    mask_0[i, :session_0_length] = 1.0
                    
                
     
            
            res = {}
           
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            
            res["users"] = np.asarray(user_list, dtype=np.int32)
            res["behaviors"] = np.asarray(item_behavior_list, dtype=np.int32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all
            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0

            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1

            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4
            
            

            res["session_count_mask"] = mask_all
            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2

             


        return res

         
    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            #self.index: data_dict["index"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.behaviors:data_dict["behaviors"],
            self.item_sequence: data_dict["item_sequence"],
            self.item_cate_sequence: data_dict["item_cate_sequence"],
            self.item_sequence_pre: data_dict["item_sequence_pre"],
            self.item_cate_sequence_pre: data_dict["item_cate_sequence_pre"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"],
            self.sequence_mask : data_dict["sequence_mask"],
            self.item_session_0: data_dict["item_session_0"],
            self.item_cate_session_0: data_dict["item_cate_session_0"],
            self.session_0_mask : data_dict["session_0_mask"],
            self.item_session_1: data_dict["item_session_1"],
            self.item_cate_session_1: data_dict["item_cate_session_1"],
            self.session_1_mask : data_dict["session_1_mask"],
            self.item_session_2: data_dict["item_session_2"],
            self.item_cate_session_2: data_dict["item_cate_session_2"],
            self.session_2_mask : data_dict["session_2_mask"],
            self.item_session_3: data_dict["item_session_3"],
            self.item_cate_session_3: data_dict["item_cate_session_3"],
            self.session_3_mask : data_dict["session_3_mask"],
            self.item_session_4: data_dict["item_session_4"],
            self.item_cate_session_4: data_dict["item_cate_session_4"],
            self.session_4_mask : data_dict["session_4_mask"],

            self.session_count_mask:data_dict["session_count_mask"],

            self.item_current_session_pre: data_dict["item_current_session_pre"],
            self.item_cate_current_session_pre: data_dict["item_cate_current_session_pre"],
            self.current_session_pre_mask : data_dict["current_session_pre_mask"],

            self.item_current_session_pre_1: data_dict["item_current_session_pre_1"],
            self.item_cate_current_session_pre_1: data_dict["item_cate_current_session_pre_1"],
            self.current_session_pre_mask_1 : data_dict["current_session_pre_mask_1"],
            self.item_current_session_pre_2: data_dict["item_current_session_pre_2"],
            self.item_cate_current_session_pre_2: data_dict["item_cate_current_session_pre_2"],
            self.current_session_pre_mask_2 : data_dict["current_session_pre_mask_2"],


            
           
        }
        return feed_dict


class SequenceSessionSuccExLSIterator(SequenceSessionSuccLSIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        super(SequenceSessionSuccExLSIterator, self).__init__(hparams, graph, col_spliter)
        self.augment_rate =hparams.augment_rate
        with self.graph.as_default():
            self.item_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_1"
            )
            self.item_cate_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_1"
            )
            self.current_session_pre_mask_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_1"
            )

            self.item_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_2"
            )
            self.item_cate_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_2"
            )
            self.current_session_pre_mask_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_2"
            )

    
    def _convert_data(
        self,
       label_list,
       
        user_list,
        item_list,
        item_cate_list,
        item_ts_list,
        item_behavior_list,
        
        item_sequence_batch,
        item_cate_sequence_batch,
        time_diff_batch,
        time_from_first_action_batch,
        time_to_now_batch,
        item_session_0_batch ,
        item_cate_session_0_batch ,
        item_session_1_batch ,
        item_cate_session_1_batch ,
        item_session_2_batch ,
        item_cate_session_2_batch, 
        item_session_3_batch ,
        item_cate_session_3_batch ,
        item_session_4_batch ,
        item_cate_session_4_batch ,

        valid_sess_batch ,
        item_sequence_pre_batch ,
        item_cate_sequence_pre_batch ,
        item_current_session_pre_batch,
        item_cate_current_session_pre_batch,
        item_current_session_succ_batch,
        item_cate_current_session_succ_batch ,
        batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            time_list (list): a list of current timestamp.
            time_diff_list (list): a list of timestamp between each sequential opertions.
            time_from_first_action_list (list): a list of timestamp from the first opertion.
            time_to_now_list (list): a list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return
            # 标量 变量
            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            #标量 不变量
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            
            behavior_list_all = np.asarray(
                [[ behavior] * (batch_num_ngs + 1) for  behavior in  item_behavior_list], dtype=np.int32
            ).flatten()
        

            # history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]#标量 不变量，后面应该要扩展
            max_seq_length  = self.max_seq_length
            max_session_count= self.max_session_count#4
            # 向量 list 不变量 先扩展位置，填充0

            item_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 



            time_diff_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_to_now_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")



            item_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            
            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k)
            ).astype("int32") 

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            

            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 



            #mask也是list
            mask_all = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_session_count )
            ).astype("float32")
            mask_sequence = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            mask_0 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_3 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_4 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            mask_pre =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            

            mask_pre_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_pre_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            #先处理历史序列
            
            for i in range(instance_cnt):
                #已经做过截断了，这里直接用
                    
                #sequence 
                sequence_length = len(item_sequence_batch[i])
                sequence_pre_length = len(item_sequence_pre_batch[i])
                for index in range(batch_num_ngs + 1):
                    item_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_batch[i] , dtype=np.int32)
                    item_cate_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_batch[i] , dtype=np.int32
                    )
                    mask_sequence[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    time_diff_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_diff_batch[i] , dtype=np.float32)
                    time_from_first_action_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        time_from_first_action_batch[i] , dtype=np.float32
                    )
                    time_to_now_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_to_now_batch[i] , dtype=np.float32)


                    item_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length 
                    ] = np.asarray(item_sequence_pre_batch[i] , dtype=np.int32)
                    item_cate_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length
                    ] = np.asarray(
                        item_cate_sequence_pre_batch[i] , dtype=np.int32
                    )
                #current_session

                current_session_pre_length = len(item_current_session_pre_batch[i])
                current_session_succ_length = len(item_current_session_succ_batch[i])
                #data AUgmentation
                #一部分交换，一部分mask
                replace_index_1 = random.sample(list(range(current_session_pre_length)),k=int(np.ceil(self.augment_rate*current_session_pre_length)))
                replace_index_2 = random.sample(list(range(current_session_succ_length)),k=int(np.ceil(self.augment_rate*current_session_succ_length)))

                
                item_current_session_pre_1 =  [item_current_session_pre_batch[i][index] for index in range(current_session_pre_length)  if index not in replace_index_1] +[item_current_session_succ_batch[i][index] for index in range(current_session_succ_length) if index  in replace_index_2]
                item_cate_current_session_pre_1 =  [item_cate_current_session_pre_batch[i][index] for index in range(current_session_pre_length) if index not in replace_index_1] +[item_cate_current_session_succ_batch[i][index] for index in range(current_session_succ_length) if index  in replace_index_2]

                item_current_session_pre_2=  [item_current_session_succ_batch[i][index] for index in range(current_session_succ_length) if index not in replace_index_2] +[item_current_session_pre_batch[i][index] for index in range(current_session_pre_length) if index  in replace_index_1]
                item_cate_current_session_pre_2 =  [item_cate_current_session_succ_batch[i][index] for index in range(current_session_succ_length) if index not in replace_index_2]+[item_cate_current_session_pre_batch[i][index] for index in range(current_session_pre_length) if index  in replace_index_1]
                current_session_pre_1_length = len(item_current_session_pre_1 )
                current_session_pre_2_length = len(item_current_session_pre_2)
                
                # #再mask
                # remove_index_1 = random.sample(list(range(current_session_pre_1_length )),k=int(np.ceil(self.augment_rate*current_session_pre_1_length)))
                # remove_index_2 = random.sample(list(range(current_session_pre_2_length)),k=int(np.ceil(self.augment_rate*current_session_pre_2_length)))



                
                
                
                
                for index in range(batch_num_ngs + 1):
                    item_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length ] = np.asarray(item_current_session_pre_batch[i] , dtype=np.int32)
                    item_cate_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = np.asarray(item_cate_current_session_pre_batch[i] , dtype=np.int32)
                    mask_pre[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = 1.0

                    item_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length ] = np.asarray(item_current_session_pre_1 , dtype=np.int32)
                    item_cate_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = np.asarray(item_cate_current_session_pre_1 , dtype=np.int32)
                    mask_pre_1[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = 1.0

                    item_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length ] = np.asarray(item_current_session_pre_2 , dtype=np.int32)
                    item_cate_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = np.asarray(item_cate_current_session_pre_2 , dtype=np.int32)
                    mask_pre_2[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = 1.0

                    
                     

                #long-term session
                #5个长期序列， 不一定存在
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count>0:
                    for index in range(batch_num_ngs + 1):
                        mask_all[i * (batch_num_ngs + 1) + index,  :sess_count ] = 1.0 
                session_4_length =valid_sess_batch[i][0]
                if session_4_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length ] = np.asarray(item_session_4_batch[i] , dtype=np.int32)
                        item_cate_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length] = np.asarray(item_cate_session_4_batch[i] , dtype=np.int32)
                        mask_4[i * (batch_num_ngs + 1) + index, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length =valid_sess_batch[i][1]
                if session_3_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length ] = np.asarray(item_session_3_batch[i] , dtype=np.int32)
                        item_cate_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length] = np.asarray(item_cate_session_3_batch[i] , dtype=np.int32)
                        mask_3[i * (batch_num_ngs + 1) + index, :session_3_length] = 1.0
                else:
                    continue
                session_2_length =valid_sess_batch[i][2]
                if session_2_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length ] = np.asarray(item_session_2_batch[i] , dtype=np.int32)
                        item_cate_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length] = np.asarray(item_cate_session_2_batch[i] , dtype=np.int32)
                        mask_2[i * (batch_num_ngs + 1) + index, :session_2_length] = 1.0
                else:
                    continue
                session_1_length =valid_sess_batch[i][3]
                if session_1_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length ] = np.asarray(item_session_1_batch[i] , dtype=np.int32)
                        item_cate_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length] = np.asarray(item_cate_session_1_batch[i] , dtype=np.int32)
                        mask_1[i * (batch_num_ngs + 1) + index, :session_1_length] = 1.0
                else:
                    continue
                session_0_length =valid_sess_batch[i][4]
                if session_0_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length ] = np.asarray(item_session_0_batch[i] , dtype=np.int32)
                        item_cate_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length] = np.asarray(item_cate_session_0_batch[i] , dtype=np.int32)
                        mask_0[i * (batch_num_ngs + 1) + index, :session_0_length] = 1.0
                


            #处理负采样
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
           
            
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
             
            res["users"] = np.asarray(user_list_all, dtype=np.int32)
            res["behaviors"] = np.asarray(behavior_list_all, dtype=np.int32)
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            

            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all

            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0
            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1
            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4

            res["session_count_mask"] = mask_all

            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2


             


        else:
            #测试和验证
            instance_cnt = len(label_list)
             
            max_seq_length  = self.max_seq_length
            max_session_count =self.max_session_count


            item_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")

            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt  , self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt  ,  self.recent_k )
            ).astype("int32")

            time_diff_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt, max_seq_length* max_session_count  )
            ).astype("float32")
            time_to_now_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            mask_sequence = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")    


            item_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_0 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_1= np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_2 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_3 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_4 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            mask_all = np.zeros((instance_cnt, max_session_count )).astype("float32")

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_1 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_2 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


             

           

   
            for i in range(instance_cnt):
                #sequence
                sequence_length = len(item_sequence_batch[i])
                item_sequence_batch_all[i, :sequence_length] = item_sequence_batch[i] 
                item_cate_sequence_batch_all[i, :sequence_length] = item_cate_sequence_batch[i] 
                sequence_pre_length = len(item_sequence_pre_batch[i])
                item_sequence_pre_batch_all[i, :sequence_pre_length] = item_sequence_pre_batch[i] 
                item_cate_sequence_pre_batch_all[i, :sequence_pre_length] = item_cate_sequence_pre_batch[i] 

                mask_sequence[i, :sequence_length] = 1.0
                time_diff_batch_all[i, :sequence_length] = time_diff_batch[i] 
                time_from_first_action_batch_all[i, :sequence_length] = time_from_first_action_batch[i] 
                time_to_now_batch_all[i, :sequence_length] = time_to_now_batch[i] 
                #succ 无需处理，保持为0即可
                current_session_pre_length = len(item_current_session_pre_batch[i])
                item_current_session_pre_batch_all[i  , :current_session_pre_length ] =  item_current_session_pre_batch[i]  
                item_cate_current_session_pre_batch_all[i  , :current_session_pre_length] =  item_cate_current_session_pre_batch[i]  
                mask_pre[i , :current_session_pre_length] = 1.0
                

                #session
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count >0 :  
                    mask_all[i, :sess_count ] = 1.0

                session_4_length = valid_sess_batch[i][0] 
                if session_4_length >0:
                  
                    item_session_4_batch_all[i, :session_4_length] = item_session_4_batch[i] 
                    item_cate_session_4_batch_all[i, :session_4_length] = item_cate_session_4_batch[i] 
                    mask_4[i, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length = valid_sess_batch[i][1] 
                if session_3_length >0:
                    
                    item_session_3_batch_all[i, :session_3_length] = item_session_3_batch[i] 
                    item_cate_session_3_batch_all[i, :session_3_length] = item_cate_session_3_batch[i] 
                    mask_3[i, :session_3_length] = 1.0
                else:
                    continue
                
                session_2_length = valid_sess_batch[i][2] 
                if session_2_length >0:
                     
                    item_session_2_batch_all[i, :session_2_length] = item_session_2_batch[i] 
                    item_cate_session_2_batch_all[i, :session_2_length] = item_cate_session_2_batch[i] 
                    mask_2[i, :session_2_length] = 1.0
                else:
                    continue
                session_1_length = valid_sess_batch[i][3] 
                if session_1_length >0:
                     
                    item_session_1_batch_all[i, :session_1_length] = item_session_1_batch[i] 
                    item_cate_session_1_batch_all[i, :session_1_length] = item_cate_session_1_batch[i] 
                    mask_1[i, :session_1_length] = 1.0
                else:
                    continue


                session_0_length = valid_sess_batch[i][4] 
                if session_0_length >0:
            
                    item_session_0_batch_all[i, :session_0_length] = item_session_0_batch[i] 
                    item_cate_session_0_batch_all[i, :session_0_length] = item_cate_session_0_batch[i] 
                    mask_0[i, :session_0_length] = 1.0
                    
                
     
            
            res = {}
           
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            
            res["users"] = np.asarray(user_list, dtype=np.int32)
            res["behaviors"] = np.asarray(item_behavior_list, dtype=np.int32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all
            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0

            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1

            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4
            
            

            res["session_count_mask"] = mask_all
            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2

             


        return res

         
    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            #self.index: data_dict["index"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.behaviors:data_dict["behaviors"],
            self.item_sequence: data_dict["item_sequence"],
            self.item_cate_sequence: data_dict["item_cate_sequence"],
            self.item_sequence_pre: data_dict["item_sequence_pre"],
            self.item_cate_sequence_pre: data_dict["item_cate_sequence_pre"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"],
            self.sequence_mask : data_dict["sequence_mask"],
            self.item_session_0: data_dict["item_session_0"],
            self.item_cate_session_0: data_dict["item_cate_session_0"],
            self.session_0_mask : data_dict["session_0_mask"],
            self.item_session_1: data_dict["item_session_1"],
            self.item_cate_session_1: data_dict["item_cate_session_1"],
            self.session_1_mask : data_dict["session_1_mask"],
            self.item_session_2: data_dict["item_session_2"],
            self.item_cate_session_2: data_dict["item_cate_session_2"],
            self.session_2_mask : data_dict["session_2_mask"],
            self.item_session_3: data_dict["item_session_3"],
            self.item_cate_session_3: data_dict["item_cate_session_3"],
            self.session_3_mask : data_dict["session_3_mask"],
            self.item_session_4: data_dict["item_session_4"],
            self.item_cate_session_4: data_dict["item_cate_session_4"],
            self.session_4_mask : data_dict["session_4_mask"],

            self.session_count_mask:data_dict["session_count_mask"],

            self.item_current_session_pre: data_dict["item_current_session_pre"],
            self.item_cate_current_session_pre: data_dict["item_cate_current_session_pre"],
            self.current_session_pre_mask : data_dict["current_session_pre_mask"],

            self.item_current_session_pre_1: data_dict["item_current_session_pre_1"],
            self.item_cate_current_session_pre_1: data_dict["item_cate_current_session_pre_1"],
            self.current_session_pre_mask_1 : data_dict["current_session_pre_mask_1"],
            self.item_current_session_pre_2: data_dict["item_current_session_pre_2"],
            self.item_cate_current_session_pre_2: data_dict["item_cate_current_session_pre_2"],
            self.current_session_pre_mask_2 : data_dict["current_session_pre_mask_2"],


            
           
        }
        return feed_dict

 


class SequenceSessionCropLSIterator(SequenceSessionSuccLSIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        super(SequenceSessionCropLSIterator, self).__init__(hparams, graph, col_spliter)
        self.augment_rate =hparams.augment_rate
        with self.graph.as_default():
            self.item_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_1"
            )
            self.item_cate_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_1"
            )
            self.current_session_pre_mask_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_1"
            )

            self.item_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_2"
            )
            self.item_cate_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_2"
            )
            self.current_session_pre_mask_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_2"
            )

    
    def _convert_data(
        self,
       label_list,
       
        user_list,
        item_list,
        item_cate_list,
        item_ts_list,
        item_behavior_list,
        
        item_sequence_batch,
        item_cate_sequence_batch,
        time_diff_batch,
        time_from_first_action_batch,
        time_to_now_batch,
        item_session_0_batch ,
        item_cate_session_0_batch ,
        item_session_1_batch ,
        item_cate_session_1_batch ,
        item_session_2_batch ,
        item_cate_session_2_batch, 
        item_session_3_batch ,
        item_cate_session_3_batch ,
        item_session_4_batch ,
        item_cate_session_4_batch ,

        valid_sess_batch ,
        item_sequence_pre_batch ,
        item_cate_sequence_pre_batch ,
        item_current_session_pre_batch,
        item_cate_current_session_pre_batch,
        item_current_session_succ_batch,
        item_cate_current_session_succ_batch ,
        batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            time_list (list): a list of current timestamp.
            time_diff_list (list): a list of timestamp between each sequential opertions.
            time_from_first_action_list (list): a list of timestamp from the first opertion.
            time_to_now_list (list): a list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return
            # 标量 变量
            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            #标量 不变量
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            
            behavior_list_all = np.asarray(
                [[ behavior] * (batch_num_ngs + 1) for  behavior in  item_behavior_list], dtype=np.int32
            ).flatten()
        

            # history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]#标量 不变量，后面应该要扩展
            max_seq_length  = self.max_seq_length
            max_session_count= self.max_session_count#4
            # 向量 list 不变量 先扩展位置，填充0

            item_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 



            time_diff_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_to_now_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")



            item_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            
            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k)
            ).astype("int32") 

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            

            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 



            #mask也是list
            mask_all = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_session_count )
            ).astype("float32")
            mask_sequence = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            mask_0 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_3 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_4 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            mask_pre =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            

            mask_pre_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_pre_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            #先处理历史序列
            
            for i in range(instance_cnt):
                #已经做过截断了，这里直接用
                    
                #sequence 
                sequence_length = len(item_sequence_batch[i])
                sequence_pre_length = len(item_sequence_pre_batch[i])
                for index in range(batch_num_ngs + 1):
                    item_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_batch[i] , dtype=np.int32)
                    item_cate_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_batch[i] , dtype=np.int32
                    )
                    mask_sequence[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    time_diff_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_diff_batch[i] , dtype=np.float32)
                    time_from_first_action_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        time_from_first_action_batch[i] , dtype=np.float32
                    )
                    time_to_now_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_to_now_batch[i] , dtype=np.float32)


                    item_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length 
                    ] = np.asarray(item_sequence_pre_batch[i] , dtype=np.int32)
                    item_cate_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length
                    ] = np.asarray(
                        item_cate_sequence_pre_batch[i] , dtype=np.int32
                    )
                #current_session
               
 
                current_session_pre_length = len(item_current_session_pre_batch[i])
                #data AUgmentation
                if current_session_pre_length>2:
                    
                    crop_length = int(np.ceil(self.augment_rate*current_session_pre_length))

                    start_index_1 = random.sample(list(range(current_session_pre_length-crop_length+1)) ,k=1)
                    start_index_2 = random.sample(list(range(current_session_pre_length-crop_length+1)),k=1 )
                    
                    item_current_session_pre_1 = item_current_session_pre_batch[i].copy()
                    del item_current_session_pre_1[start_index_1[0]:start_index_1[0]+crop_length]#包括起始位
                    item_cate_current_session_pre_1 =  item_cate_current_session_pre_batch[i].copy()
                    del item_cate_current_session_pre_1[start_index_1[0]:start_index_1[0]+crop_length]

                    item_current_session_pre_2 = item_current_session_pre_batch[i].copy()
                    del item_current_session_pre_2[start_index_2[0]:start_index_2[0]+crop_length]

                    item_cate_current_session_pre_2 =  item_cate_current_session_pre_batch[i].copy()
                    del item_cate_current_session_pre_2[start_index_2[0]:start_index_2[0]+crop_length]

                    current_session_pre_1_length = len(item_current_session_pre_1 )
                    current_session_pre_2_length = len(item_current_session_pre_2)
                else:
                    #太短了就不裁了
                     
                    item_current_session_pre_1 =   item_current_session_pre_batch[i] 
                    item_cate_current_session_pre_1 =  item_cate_current_session_pre_batch[i]  

                    item_current_session_pre_2=   item_current_session_pre_batch[i] 
                    item_cate_current_session_pre_2 =  item_cate_current_session_pre_batch[i]  

                    current_session_pre_1_length = len(item_current_session_pre_1 )
                    current_session_pre_2_length = len(item_current_session_pre_2)

                
                
                
                
                for index in range(batch_num_ngs + 1):
                    item_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length ] = np.asarray(item_current_session_pre_batch[i] , dtype=np.int32)
                    item_cate_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = np.asarray(item_cate_current_session_pre_batch[i] , dtype=np.int32)
                    mask_pre[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = 1.0

                    item_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length ] = np.asarray(item_current_session_pre_1 , dtype=np.int32)
                    item_cate_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = np.asarray(item_cate_current_session_pre_1 , dtype=np.int32)
                    mask_pre_1[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = 1.0

                    item_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length ] = np.asarray(item_current_session_pre_2 , dtype=np.int32)
                    item_cate_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = np.asarray(item_cate_current_session_pre_2 , dtype=np.int32)
                    mask_pre_2[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = 1.0

                    
                     

                #long-term session
                #5个长期序列， 不一定存在
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count>0:
                    for index in range(batch_num_ngs + 1):
                        mask_all[i * (batch_num_ngs + 1) + index,  :sess_count ] = 1.0 
                session_4_length =valid_sess_batch[i][0]
                if session_4_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length ] = np.asarray(item_session_4_batch[i] , dtype=np.int32)
                        item_cate_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length] = np.asarray(item_cate_session_4_batch[i] , dtype=np.int32)
                        mask_4[i * (batch_num_ngs + 1) + index, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length =valid_sess_batch[i][1]
                if session_3_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length ] = np.asarray(item_session_3_batch[i] , dtype=np.int32)
                        item_cate_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length] = np.asarray(item_cate_session_3_batch[i] , dtype=np.int32)
                        mask_3[i * (batch_num_ngs + 1) + index, :session_3_length] = 1.0
                else:
                    continue
                session_2_length =valid_sess_batch[i][2]
                if session_2_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length ] = np.asarray(item_session_2_batch[i] , dtype=np.int32)
                        item_cate_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length] = np.asarray(item_cate_session_2_batch[i] , dtype=np.int32)
                        mask_2[i * (batch_num_ngs + 1) + index, :session_2_length] = 1.0
                else:
                    continue
                session_1_length =valid_sess_batch[i][3]
                if session_1_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length ] = np.asarray(item_session_1_batch[i] , dtype=np.int32)
                        item_cate_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length] = np.asarray(item_cate_session_1_batch[i] , dtype=np.int32)
                        mask_1[i * (batch_num_ngs + 1) + index, :session_1_length] = 1.0
                else:
                    continue
                session_0_length =valid_sess_batch[i][4]
                if session_0_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length ] = np.asarray(item_session_0_batch[i] , dtype=np.int32)
                        item_cate_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length] = np.asarray(item_cate_session_0_batch[i] , dtype=np.int32)
                        mask_0[i * (batch_num_ngs + 1) + index, :session_0_length] = 1.0
                


            #处理负采样
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
           
             
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
             
            res["users"] = np.asarray(user_list_all, dtype=np.int32)
            res["behaviors"] = np.asarray(behavior_list_all, dtype=np.int32)
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            

            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all

            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0
            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1
            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4

            res["session_count_mask"] = mask_all

            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2


             


        else:
            #测试和验证
            instance_cnt = len(label_list)
             
            max_seq_length  = self.max_seq_length
            max_session_count =self.max_session_count


            item_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")

            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt  , self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt  ,  self.recent_k )
            ).astype("int32")

            time_diff_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt, max_seq_length* max_session_count  )
            ).astype("float32")
            time_to_now_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            mask_sequence = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")    


            item_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_0 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_1= np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_2 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_3 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_4 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            mask_all = np.zeros((instance_cnt, max_session_count )).astype("float32")

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_1 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_2 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


             

           

   
            for i in range(instance_cnt):
                #sequence
                sequence_length = len(item_sequence_batch[i])
                item_sequence_batch_all[i, :sequence_length] = item_sequence_batch[i] 
                item_cate_sequence_batch_all[i, :sequence_length] = item_cate_sequence_batch[i] 
                sequence_pre_length = len(item_sequence_pre_batch[i])
                item_sequence_pre_batch_all[i, :sequence_pre_length] = item_sequence_pre_batch[i] 
                item_cate_sequence_pre_batch_all[i, :sequence_pre_length] = item_cate_sequence_pre_batch[i] 

                mask_sequence[i, :sequence_length] = 1.0
                time_diff_batch_all[i, :sequence_length] = time_diff_batch[i] 
                time_from_first_action_batch_all[i, :sequence_length] = time_from_first_action_batch[i] 
                time_to_now_batch_all[i, :sequence_length] = time_to_now_batch[i] 
                #succ 无需处理，保持为0即可
                current_session_pre_length = len(item_current_session_pre_batch[i])
                item_current_session_pre_batch_all[i  , :current_session_pre_length ] =  item_current_session_pre_batch[i]  
                item_cate_current_session_pre_batch_all[i  , :current_session_pre_length] =  item_cate_current_session_pre_batch[i]  
                mask_pre[i , :current_session_pre_length] = 1.0
                

                #session
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count >0 :  
                    mask_all[i, :sess_count ] = 1.0

                session_4_length = valid_sess_batch[i][0] 
                if session_4_length >0:
                  
                    item_session_4_batch_all[i, :session_4_length] = item_session_4_batch[i] 
                    item_cate_session_4_batch_all[i, :session_4_length] = item_cate_session_4_batch[i] 
                    mask_4[i, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length = valid_sess_batch[i][1] 
                if session_3_length >0:
                    
                    item_session_3_batch_all[i, :session_3_length] = item_session_3_batch[i] 
                    item_cate_session_3_batch_all[i, :session_3_length] = item_cate_session_3_batch[i] 
                    mask_3[i, :session_3_length] = 1.0
                else:
                    continue
                
                session_2_length = valid_sess_batch[i][2] 
                if session_2_length >0:
                     
                    item_session_2_batch_all[i, :session_2_length] = item_session_2_batch[i] 
                    item_cate_session_2_batch_all[i, :session_2_length] = item_cate_session_2_batch[i] 
                    mask_2[i, :session_2_length] = 1.0
                else:
                    continue
                session_1_length = valid_sess_batch[i][3] 
                if session_1_length >0:
                     
                    item_session_1_batch_all[i, :session_1_length] = item_session_1_batch[i] 
                    item_cate_session_1_batch_all[i, :session_1_length] = item_cate_session_1_batch[i] 
                    mask_1[i, :session_1_length] = 1.0
                else:
                    continue


                session_0_length = valid_sess_batch[i][4] 
                if session_0_length >0:
            
                    item_session_0_batch_all[i, :session_0_length] = item_session_0_batch[i] 
                    item_cate_session_0_batch_all[i, :session_0_length] = item_cate_session_0_batch[i] 
                    mask_0[i, :session_0_length] = 1.0
                    
                
     
            
            res = {}
           
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            
            res["users"] = np.asarray(user_list, dtype=np.int32)
            res["behaviors"] = np.asarray(item_behavior_list, dtype=np.int32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all
            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0

            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1

            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4
            
            

            res["session_count_mask"] = mask_all
            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2

             


        return res

         
    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            #self.index: data_dict["index"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.behaviors:data_dict["behaviors"],
            self.item_sequence: data_dict["item_sequence"],
            self.item_cate_sequence: data_dict["item_cate_sequence"],
            self.item_sequence_pre: data_dict["item_sequence_pre"],
            self.item_cate_sequence_pre: data_dict["item_cate_sequence_pre"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"],
            self.sequence_mask : data_dict["sequence_mask"],
            self.item_session_0: data_dict["item_session_0"],
            self.item_cate_session_0: data_dict["item_cate_session_0"],
            self.session_0_mask : data_dict["session_0_mask"],
            self.item_session_1: data_dict["item_session_1"],
            self.item_cate_session_1: data_dict["item_cate_session_1"],
            self.session_1_mask : data_dict["session_1_mask"],
            self.item_session_2: data_dict["item_session_2"],
            self.item_cate_session_2: data_dict["item_cate_session_2"],
            self.session_2_mask : data_dict["session_2_mask"],
            self.item_session_3: data_dict["item_session_3"],
            self.item_cate_session_3: data_dict["item_cate_session_3"],
            self.session_3_mask : data_dict["session_3_mask"],
            self.item_session_4: data_dict["item_session_4"],
            self.item_cate_session_4: data_dict["item_cate_session_4"],
            self.session_4_mask : data_dict["session_4_mask"],

            self.session_count_mask:data_dict["session_count_mask"],

            self.item_current_session_pre: data_dict["item_current_session_pre"],
            self.item_cate_current_session_pre: data_dict["item_cate_current_session_pre"],
            self.current_session_pre_mask : data_dict["current_session_pre_mask"],

            self.item_current_session_pre_1: data_dict["item_current_session_pre_1"],
            self.item_cate_current_session_pre_1: data_dict["item_cate_current_session_pre_1"],
            self.current_session_pre_mask_1 : data_dict["current_session_pre_mask_1"],
            self.item_current_session_pre_2: data_dict["item_current_session_pre_2"],
            self.item_cate_current_session_pre_2: data_dict["item_cate_current_session_pre_2"],
            self.current_session_pre_mask_2 : data_dict["current_session_pre_mask_2"],


            
           
        }
        return feed_dict


class SequenceSessionReorderLSIterator(SequenceSessionSuccLSIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        super(SequenceSessionReorderLSIterator, self).__init__(hparams, graph, col_spliter)
        self.augment_rate =hparams.augment_rate
        with self.graph.as_default():
            self.item_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_1"
            )
            self.item_cate_current_session_pre_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_1"
            )
            self.current_session_pre_mask_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_1"
            )

            self.item_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_current_session_pre_2"
            )
            self.item_cate_current_session_pre_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_current_session_pre_2"
            )
            self.current_session_pre_mask_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="current_session_pre_mask_2"
            )

    
    def _convert_data(
        self,
       label_list,
       
        user_list,
        item_list,
        item_cate_list,
        item_ts_list,
        item_behavior_list,
        
        item_sequence_batch,
        item_cate_sequence_batch,
        time_diff_batch,
        time_from_first_action_batch,
        time_to_now_batch,
        item_session_0_batch ,
        item_cate_session_0_batch ,
        item_session_1_batch ,
        item_cate_session_1_batch ,
        item_session_2_batch ,
        item_cate_session_2_batch, 
        item_session_3_batch ,
        item_cate_session_3_batch ,
        item_session_4_batch ,
        item_cate_session_4_batch ,

        valid_sess_batch ,
        item_sequence_pre_batch ,
        item_cate_sequence_pre_batch ,
        item_current_session_pre_batch,
        item_cate_current_session_pre_batch,
        item_current_session_succ_batch,
        item_cate_current_session_succ_batch ,
        batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            time_list (list): a list of current timestamp.
            time_diff_list (list): a list of timestamp between each sequential opertions.
            time_from_first_action_list (list): a list of timestamp from the first opertion.
            time_to_now_list (list): a list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return
            # 标量 变量
            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            #标量 不变量
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            
            behavior_list_all = np.asarray(
                [[ behavior] * (batch_num_ngs + 1) for  behavior in  item_behavior_list], dtype=np.int32
            ).flatten()
        

            # history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]#标量 不变量，后面应该要扩展
            max_seq_length  = self.max_seq_length
            max_session_count= self.max_session_count#4
            # 向量 list 不变量 先扩展位置，填充0

            item_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 



            time_diff_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")
            time_to_now_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("float32")



            item_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32")
            
            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), self.recent_k)
            ).astype("int32") 

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            

            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length )
            ).astype("int32") 



            #mask也是list
            mask_all = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_session_count )
            ).astype("float32")
            mask_sequence = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            mask_0 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_3 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_4 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            mask_pre =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            

            mask_pre_1 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")
            mask_pre_2 =np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length )
            ).astype("float32")

            #先处理历史序列
            
            for i in range(instance_cnt):
                #已经做过截断了，这里直接用
                    
                #sequence 
                sequence_length = len(item_sequence_batch[i])
                sequence_pre_length = len(item_sequence_pre_batch[i])
                for index in range(batch_num_ngs + 1):
                    item_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_batch[i] , dtype=np.int32)
                    item_cate_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_batch[i] , dtype=np.int32
                    )
                    mask_sequence[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    time_diff_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_diff_batch[i] , dtype=np.float32)
                    time_from_first_action_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        time_from_first_action_batch[i] , dtype=np.float32
                    )
                    time_to_now_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(time_to_now_batch[i] , dtype=np.float32)


                    item_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length 
                    ] = np.asarray(item_sequence_pre_batch[i] , dtype=np.int32)
                    item_cate_sequence_pre_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_pre_length
                    ] = np.asarray(
                        item_cate_sequence_pre_batch[i] , dtype=np.int32
                    )
                #current_session
               
 
                current_session_pre_length = len(item_current_session_pre_batch[i])
                #data AUgmentation
                if current_session_pre_length>2:
                    
                    crop_length = int(np.ceil(self.augment_rate*current_session_pre_length))

                    start_index_1 = random.sample(list(range(current_session_pre_length-crop_length+1)) ,k=1)
                    start_index_2 = random.sample(list(range(current_session_pre_length-crop_length+1)),k=1 )
                    

                    current_session_pre_1 =np.array([ item_current_session_pre_batch[i].copy(), item_cate_current_session_pre_batch[i].copy()])
                    current_session_pre_1 = np.transpose(current_session_pre_1)
                    np.random.shuffle(current_session_pre_1[start_index_1[0]:start_index_1[0]+crop_length])
                    current_session_pre_1 = np.transpose(current_session_pre_1)


                    item_current_session_pre_1 = current_session_pre_1[0].tolist()
                    item_cate_current_session_pre_1 =  current_session_pre_1[1].tolist()
                     

                    current_session_pre_2 =np.array([ item_current_session_pre_batch[i].copy(), item_cate_current_session_pre_batch[i].copy()])
                    current_session_pre_2 = np.transpose(current_session_pre_2)
                    np.random.shuffle(current_session_pre_2[start_index_2[0]:start_index_2[0]+crop_length])
                    current_session_pre_2 = np.transpose(current_session_pre_2)

                    item_current_session_pre_2 = current_session_pre_2[0].tolist()
                    
                    item_cate_current_session_pre_2 =  current_session_pre_2[1].tolist()
                     

                    current_session_pre_1_length = len(item_current_session_pre_1 )
                    current_session_pre_2_length = len(item_current_session_pre_2)
                else:
                    #太短了就不裁了
                    
                    item_current_session_pre_1 =   item_current_session_pre_batch[i] 
                    item_cate_current_session_pre_1 =  item_cate_current_session_pre_batch[i]  

                    item_current_session_pre_2=   item_current_session_pre_batch[i] 
                    item_cate_current_session_pre_2 =  item_cate_current_session_pre_batch[i]  

                    current_session_pre_1_length = len(item_current_session_pre_1 )
                    current_session_pre_2_length = len(item_current_session_pre_2)

                
                
                
                
                for index in range(batch_num_ngs + 1):
                    item_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length ] = np.asarray(item_current_session_pre_batch[i] , dtype=np.int32)
                    item_cate_current_session_pre_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = np.asarray(item_cate_current_session_pre_batch[i] , dtype=np.int32)
                    mask_pre[i * (batch_num_ngs + 1) + index, :current_session_pre_length] = 1.0

                    item_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length ] = np.asarray(item_current_session_pre_1 , dtype=np.int32)
                    item_cate_current_session_pre_1_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = np.asarray(item_cate_current_session_pre_1 , dtype=np.int32)
                    mask_pre_1[i * (batch_num_ngs + 1) + index, :current_session_pre_1_length] = 1.0

                    item_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length ] = np.asarray(item_current_session_pre_2 , dtype=np.int32)
                    item_cate_current_session_pre_2_batch_all[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = np.asarray(item_cate_current_session_pre_2 , dtype=np.int32)
                    mask_pre_2[i * (batch_num_ngs + 1) + index, :current_session_pre_2_length] = 1.0

                    
                     

                #long-term session
                #5个长期序列， 不一定存在
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count>0:
                    for index in range(batch_num_ngs + 1):
                        mask_all[i * (batch_num_ngs + 1) + index,  :sess_count ] = 1.0 
                session_4_length =valid_sess_batch[i][0]
                if session_4_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length ] = np.asarray(item_session_4_batch[i] , dtype=np.int32)
                        item_cate_session_4_batch_all[i * (batch_num_ngs + 1) + index, :session_4_length] = np.asarray(item_cate_session_4_batch[i] , dtype=np.int32)
                        mask_4[i * (batch_num_ngs + 1) + index, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length =valid_sess_batch[i][1]
                if session_3_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length ] = np.asarray(item_session_3_batch[i] , dtype=np.int32)
                        item_cate_session_3_batch_all[i * (batch_num_ngs + 1) + index, :session_3_length] = np.asarray(item_cate_session_3_batch[i] , dtype=np.int32)
                        mask_3[i * (batch_num_ngs + 1) + index, :session_3_length] = 1.0
                else:
                    continue
                session_2_length =valid_sess_batch[i][2]
                if session_2_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length ] = np.asarray(item_session_2_batch[i] , dtype=np.int32)
                        item_cate_session_2_batch_all[i * (batch_num_ngs + 1) + index, :session_2_length] = np.asarray(item_cate_session_2_batch[i] , dtype=np.int32)
                        mask_2[i * (batch_num_ngs + 1) + index, :session_2_length] = 1.0
                else:
                    continue
                session_1_length =valid_sess_batch[i][3]
                if session_1_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length ] = np.asarray(item_session_1_batch[i] , dtype=np.int32)
                        item_cate_session_1_batch_all[i * (batch_num_ngs + 1) + index, :session_1_length] = np.asarray(item_cate_session_1_batch[i] , dtype=np.int32)
                        mask_1[i * (batch_num_ngs + 1) + index, :session_1_length] = 1.0
                else:
                    continue
                session_0_length =valid_sess_batch[i][4]
                if session_0_length>0 :
                     
                    for index in range(batch_num_ngs + 1):
                        item_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length ] = np.asarray(item_session_0_batch[i] , dtype=np.int32)
                        item_cate_session_0_batch_all[i * (batch_num_ngs + 1) + index, :session_0_length] = np.asarray(item_cate_session_0_batch[i] , dtype=np.int32)
                        mask_0[i * (batch_num_ngs + 1) + index, :session_0_length] = 1.0
                


            #处理负采样
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
           
             
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
             
            res["users"] = np.asarray(user_list_all, dtype=np.int32)
            res["behaviors"] = np.asarray(behavior_list_all, dtype=np.int32)
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            

            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all

            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0
            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1
            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4

            res["session_count_mask"] = mask_all

            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2


             


        else:
            #测试和验证
            instance_cnt = len(label_list)
             
            max_seq_length  = self.max_seq_length
            max_session_count =self.max_session_count


            item_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")

            item_sequence_pre_batch_all = np.zeros(
                (instance_cnt  , self.recent_k )
            ).astype("int32")
            item_cate_sequence_pre_batch_all = np.zeros(
                (instance_cnt  ,  self.recent_k )
            ).astype("int32")

            time_diff_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            time_from_first_action_batch_all = np.zeros(
                (instance_cnt, max_seq_length* max_session_count  )
            ).astype("float32")
            time_to_now_batch_all = np.zeros((instance_cnt, max_seq_length* max_session_count  )).astype(
                "float32"
            )
            mask_sequence = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")    


            item_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_0_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_0 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_1_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_1= np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_2_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_2 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_3_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_3 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            item_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            item_cate_session_4_batch_all = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")
            mask_4 = np.zeros(
                (instance_cnt, max_seq_length )
            ).astype("int32")

            mask_all = np.zeros((instance_cnt, max_session_count )).astype("float32")

            item_current_session_pre_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


            item_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_1_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_1 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")

            item_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt   , max_seq_length  )
            ).astype("int32")
            item_cate_current_session_pre_2_batch_all = np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("int32") 

            mask_pre_2 =np.zeros(
                (instance_cnt  , max_seq_length )
            ).astype("float32")


             

           

   
            for i in range(instance_cnt):
                #sequence
                sequence_length = len(item_sequence_batch[i])
                item_sequence_batch_all[i, :sequence_length] = item_sequence_batch[i] 
                item_cate_sequence_batch_all[i, :sequence_length] = item_cate_sequence_batch[i] 
                sequence_pre_length = len(item_sequence_pre_batch[i])
                item_sequence_pre_batch_all[i, :sequence_pre_length] = item_sequence_pre_batch[i] 
                item_cate_sequence_pre_batch_all[i, :sequence_pre_length] = item_cate_sequence_pre_batch[i] 

                mask_sequence[i, :sequence_length] = 1.0
                time_diff_batch_all[i, :sequence_length] = time_diff_batch[i] 
                time_from_first_action_batch_all[i, :sequence_length] = time_from_first_action_batch[i] 
                time_to_now_batch_all[i, :sequence_length] = time_to_now_batch[i] 
                #succ 无需处理，保持为0即可
                current_session_pre_length = len(item_current_session_pre_batch[i])
                item_current_session_pre_batch_all[i  , :current_session_pre_length ] =  item_current_session_pre_batch[i]  
                item_cate_current_session_pre_batch_all[i  , :current_session_pre_length] =  item_cate_current_session_pre_batch[i]  
                mask_pre[i , :current_session_pre_length] = 1.0
                

                #session
                sess_count=len(np.nonzero(valid_sess_batch[i])[0])
                if sess_count >0 :  
                    mask_all[i, :sess_count ] = 1.0

                session_4_length = valid_sess_batch[i][0] 
                if session_4_length >0:
                  
                    item_session_4_batch_all[i, :session_4_length] = item_session_4_batch[i] 
                    item_cate_session_4_batch_all[i, :session_4_length] = item_cate_session_4_batch[i] 
                    mask_4[i, :session_4_length] = 1.0
                else:
                    continue
                
                session_3_length = valid_sess_batch[i][1] 
                if session_3_length >0:
                    
                    item_session_3_batch_all[i, :session_3_length] = item_session_3_batch[i] 
                    item_cate_session_3_batch_all[i, :session_3_length] = item_cate_session_3_batch[i] 
                    mask_3[i, :session_3_length] = 1.0
                else:
                    continue
                
                session_2_length = valid_sess_batch[i][2] 
                if session_2_length >0:
                     
                    item_session_2_batch_all[i, :session_2_length] = item_session_2_batch[i] 
                    item_cate_session_2_batch_all[i, :session_2_length] = item_cate_session_2_batch[i] 
                    mask_2[i, :session_2_length] = 1.0
                else:
                    continue
                session_1_length = valid_sess_batch[i][3] 
                if session_1_length >0:
                     
                    item_session_1_batch_all[i, :session_1_length] = item_session_1_batch[i] 
                    item_cate_session_1_batch_all[i, :session_1_length] = item_cate_session_1_batch[i] 
                    mask_1[i, :session_1_length] = 1.0
                else:
                    continue


                session_0_length = valid_sess_batch[i][4] 
                if session_0_length >0:
            
                    item_session_0_batch_all[i, :session_0_length] = item_session_0_batch[i] 
                    item_cate_session_0_batch_all[i, :session_0_length] = item_cate_session_0_batch[i] 
                    mask_0[i, :session_0_length] = 1.0
                    
                
     
            
            res = {}
           
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            
            res["users"] = np.asarray(user_list, dtype=np.int32)
            res["behaviors"] = np.asarray(item_behavior_list, dtype=np.int32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_pre"] = item_sequence_pre_batch_all
            res["item_cate_sequence_pre"] = item_cate_sequence_pre_batch_all
            res["time_diff"] = time_diff_batch_all
            res["time_from_first_action"] = time_from_first_action_batch_all
            res["time_to_now"] = time_to_now_batch_all
            res["sequence_mask"] =mask_sequence
            
            res["item_session_0"] = item_session_0_batch_all
            res["item_cate_session_0"] = item_cate_session_0_batch_all
            res["session_0_mask"] =mask_0

            res["item_session_1"] = item_session_1_batch_all
            res["item_cate_session_1"] = item_cate_session_1_batch_all
            res["session_1_mask"] =mask_1

            res["item_session_2"] = item_session_2_batch_all
            res["item_cate_session_2"] = item_cate_session_2_batch_all
            res["session_2_mask"] =mask_2
            res["item_session_3"] = item_session_3_batch_all
            res["item_cate_session_3"] = item_cate_session_3_batch_all
            res["session_3_mask"] =mask_3
            res["item_session_4"] = item_session_4_batch_all
            res["item_cate_session_4"] = item_cate_session_4_batch_all
            res["session_4_mask"] =mask_4
            
            

            res["session_count_mask"] = mask_all
            res["item_current_session_pre"] = item_current_session_pre_batch_all
            res["item_cate_current_session_pre"] = item_cate_current_session_pre_batch_all
            res["current_session_pre_mask"] = mask_pre

            res["item_current_session_pre_1"] = item_current_session_pre_1_batch_all
            res["item_cate_current_session_pre_1"] = item_cate_current_session_pre_1_batch_all
            res["current_session_pre_mask_1"] = mask_pre_1

            res["item_current_session_pre_2"] = item_current_session_pre_2_batch_all
            res["item_cate_current_session_pre_2"] = item_cate_current_session_pre_2_batch_all
            res["current_session_pre_mask_2"] = mask_pre_2

             


        return res

         
    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            #self.index: data_dict["index"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.behaviors:data_dict["behaviors"],
            self.item_sequence: data_dict["item_sequence"],
            self.item_cate_sequence: data_dict["item_cate_sequence"],
            self.item_sequence_pre: data_dict["item_sequence_pre"],
            self.item_cate_sequence_pre: data_dict["item_cate_sequence_pre"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"],
            self.sequence_mask : data_dict["sequence_mask"],
            self.item_session_0: data_dict["item_session_0"],
            self.item_cate_session_0: data_dict["item_cate_session_0"],
            self.session_0_mask : data_dict["session_0_mask"],
            self.item_session_1: data_dict["item_session_1"],
            self.item_cate_session_1: data_dict["item_cate_session_1"],
            self.session_1_mask : data_dict["session_1_mask"],
            self.item_session_2: data_dict["item_session_2"],
            self.item_cate_session_2: data_dict["item_cate_session_2"],
            self.session_2_mask : data_dict["session_2_mask"],
            self.item_session_3: data_dict["item_session_3"],
            self.item_cate_session_3: data_dict["item_cate_session_3"],
            self.session_3_mask : data_dict["session_3_mask"],
            self.item_session_4: data_dict["item_session_4"],
            self.item_cate_session_4: data_dict["item_cate_session_4"],
            self.session_4_mask : data_dict["session_4_mask"],

            self.session_count_mask:data_dict["session_count_mask"],

            self.item_current_session_pre: data_dict["item_current_session_pre"],
            self.item_cate_current_session_pre: data_dict["item_cate_current_session_pre"],
            self.current_session_pre_mask : data_dict["current_session_pre_mask"],

            self.item_current_session_pre_1: data_dict["item_current_session_pre_1"],
            self.item_cate_current_session_pre_1: data_dict["item_cate_current_session_pre_1"],
            self.current_session_pre_mask_1 : data_dict["current_session_pre_mask_1"],
            self.item_current_session_pre_2: data_dict["item_current_session_pre_2"],
            self.item_cate_current_session_pre_2: data_dict["item_cate_current_session_pre_2"],
            self.current_session_pre_mask_2 : data_dict["current_session_pre_mask_2"],


            
           
        }
        return feed_dict

