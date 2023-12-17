#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import sys
sys.path.append("../../")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time

from reco_utils.common.constants import SEED
from reco_utils.recommender.deeprec.deeprec_utils import (
    prepare_hparams
)
 
 
from reco_utils.recommender.deeprec.models.sequential.rhcin import RHCINModel
 

import pdb 
 

 

from reco_utils.recommender.deeprec.io.sequence_session_w_succ_iterator import (SequenceSessionSuccExLSIterator, 
                                                                                SequenceSessionMaskLSIterator,SequenceSessionCropLSIterator,
                                                                                SequenceSessionReorderLSIterator,
                                                                        )
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'taobao', 'Dataset name.')
flags.DEFINE_integer('gpu_id', 1, 'GPU ID.')
flags.DEFINE_integer('clik_behavior', 1, 'behavior 1,2,3,4.')
flags.DEFINE_integer('max_session_count',4, 'session 0,1,2,3,4.')
flags.DEFINE_string('data_type',"session_LS/360", 'session/interval , sequence.')
flags.DEFINE_float('tau', 1.0, 'temperature coefficient.')
flags.DEFINE_float('augment_rate', 0.2, 'temperature coefficient.')
#flags.DEFINE_integer('random_seed',4, 'session 0,1,2,3,4.')
flags.DEFINE_float('alpha_margin', 0.1, 'Loss weight for contrastive of long and short intention.')
flags.DEFINE_float('start_forget_lambda', 0.1, 'Loss weight for contrastive of long and short intention.')
flags.DEFINE_float('end_forget_lambda', 0.5, 'Loss weight for contrastive of long and short intention.')
flags.DEFINE_float('alpha_loss_weight', 0.1, 'alpha_loss weight for combination of  long and short intention.')
#flags.DEFINE_list('layer_sizes', [100,64], 'alpha_loss weight for combination of  long and short intention.')
 

flags.DEFINE_integer('start_topN',300, 'session 0,1,2,3,4.')
flags.DEFINE_integer('end_topN',500, 'session 0,1,2,3,4.')
flags.DEFINE_integer('start_iteration',0, 'session 0,1,2,3,4.')
flags.DEFINE_integer('duration_iteration',12000, 'session 0,1,2,3,4.')  
flags.DEFINE_integer('valid_step', 1500, 'Step for showing metrics.')
 
flags.DEFINE_float('contrastive_loss_weight', 0.1, 'Contrastive loss, could be bpr or triplet.')
flags.DEFINE_string('contrastive_loss', 'triplet', 'Contrastive loss, could be bpr or triplet.')
flags.DEFINE_integer('contrastive_length_threshold', 5, 'Minimum sequence length value to apply contrastive loss.')
flags.DEFINE_integer('contrastive_recent_k', 3, 'Use the most recent k embeddings to compute short-term proxy.')
flags.DEFINE_integer('max_grad_norm', 2, 'Whether to clip gradient norm.')

flags.DEFINE_integer('val_num_ngs', 4, 'Number of negative instances with a positiver instance for validation.')
flags.DEFINE_integer('test_num_ngs', 99, 'Number of negative instances with a positive instance for testing.')
flags.DEFINE_integer('batch_size', 500, 'Batch size.')
flags.DEFINE_string('save_path', '', 'Save path.')
 
flags.DEFINE_string('name', 'taobao-rhcin-debug', 'Experiment name.')
flags.DEFINE_string('model', 'RHCIN', 'Model name.')
flags.DEFINE_boolean('only_test', False, 'Only test and do not train.')
flags.DEFINE_boolean('write_prediction_to_file', False, 'Whether to write prediction to file.')
 
 
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('early_stop',10, 'Patience for early stop.')
flags.DEFINE_string('data_path', os.path.join("..", "..", "tests", "resources", "deeprec", "sequential"), 'Data file path.')
flags.DEFINE_integer('train_num_ngs', 4, 'Number of negative instances with a positive instance for training.')
flags.DEFINE_integer('is_clip_norm', 1, 'Whether to clip gradient norm.')

flags.DEFINE_float('embed_l2', 1e-6, 'L2 regulation for embeddings.')
flags.DEFINE_float('layer_l2', 1e-6, 'L2 regulation for layers.')
flags.DEFINE_float('attn_loss_weight', 0.001, 'Loss weight for supervised attention.')
flags.DEFINE_float('triplet_margin', 1.0, 'Margin value for triplet loss.')
flags.DEFINE_float('discrepancy_loss_weight', 0.01, 'Loss weight for discrepancy between long and short term user embedding.')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('show_step', 500, 'Step for showing metrics.')


def get_model(flags_obj, model_path, summary_path, user_vocab, item_vocab, cate_vocab, train_num_ngs):

    EPOCHS = flags_obj.epochs
    BATCH_SIZE = flags_obj.batch_size
    #RANDOM_SEED =flags_obj.random_seed  # Set None for non-deterministic result
    RANDOM_SEED =None
   
     
    pairwise_metrics = ['mean_mrr', 'ndcg@2;5;10', 'hit@2;5;10']
    weighted_metrics = ['wauc']
    max_seq_length = 10
    time_unit = 's'
    if flags_obj.model in[ "RHCIN"]   : 
        input_creator =    SequenceSessionSuccExLSIterator
    
    elif flags_obj.model in['RHCIN_mask' ]:
        input_creator = SequenceSessionMaskLSIterator
    elif flags_obj.model in ["RHCIN_crop" ]:
        input_creator =SequenceSessionCropLSIterator
    elif flags_obj.model in ["RHCIN_reorder" ]:
        input_creator =SequenceSessionReorderLSIterator
 
    
     
     
     
    
    #SliRec
     

    if flags_obj.model in['RHCIN','RHCIN_mask',"RHCIN_crop","RHCIN_reorder" ] :
        yaml_file = '../../reco_utils/recommender/deeprec/config/rhcin.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                               # forget_lambda =flags_obj.forget_lambda,
                                tau= flags_obj.tau,
                                augment_rate=flags_obj.augment_rate,
                                #alpha_margin=flags_obj.alpha_margin,
                               # start_topN =flags_obj.start_topN ,
                               # end_topN= flags_obj.end_topN,
                               # start_iteration= flags_obj.start_iteration,
                               # duration_iteration  =flags_obj.duration_iteration ,
                                clik_behavior=flags_obj.clik_behavior,
                                max_session_count=flags_obj.max_session_count,
                                valid_step =flags_obj.valid_step,
                                contrastive_recent_k=flags_obj.contrastive_recent_k,#另有别用
                                discrepancy_loss_weight=flags_obj.discrepancy_loss_weight,
                                alpha_loss_weight=flags_obj.alpha_loss_weight,
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                contrastive_length_threshold  =flags_obj.contrastive_length_threshold,
                              contrastive_loss_weight =flags_obj.contrastive_loss_weight ,
                                is_clip_norm=flags_obj.is_clip_norm,
                             
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                
                                time_unit=time_unit,
                    )
        model = RHCINModel(hparams,  input_creator,  seed=RANDOM_SEED)
    
    

    return model


def main(argv):

    flags_obj = FLAGS

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))
 
    print('start experiment')
    data_type =flags_obj.data_type
    data_path = os.path.join(flags_obj.data_path, flags_obj.dataset,data_type)
    

     
    user_vocab = os.path.join(data_path, r'user_vocab.pkl')
    item_vocab = os.path.join(data_path, r'item_vocab.pkl')
    cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
    train_file = os.path.join(data_path, r'train_data')
    valid_file = os.path.join(data_path, r'sample_valid_data')
    test_file = os.path.join(data_path, r'sample_test_data')
   

 
    train_num_ngs = flags_obj.train_num_ngs
    valid_num_ngs = flags_obj.val_num_ngs
    test_num_ngs = flags_obj.test_num_ngs
    

     

    save_path = os.path.join(flags_obj.save_path, flags_obj.model, flags_obj.name)
    model_path = os.path.join(save_path, "model/")
    summary_path = os.path.join(save_path, "summary/")
    output_file = os.path.join(summary_path, r'output.txt')
    
    model = get_model(flags_obj, model_path, summary_path, user_vocab, item_vocab, cate_vocab, train_num_ngs)

    if flags_obj.only_test:
        ckpt_path = tf.train.latest_checkpoint(model_path)
        model.load_model(ckpt_path)
        if flags_obj.write_prediction_to_file:
            res = model.run_weighted_eval_write(test_file, output_file,num_ngs=test_num_ngs)
        else:
            res = model.run_weighted_eval(test_file, num_ngs=test_num_ngs) # test_num_ngs is the number of negative lines after each positive line in your test_file
        print(res)
        

        return

    eval_metric = 'wauc'
    
    start_time = time.time()
    model = model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric) 
    # valid_num_ngs is the number of negative lines after each positive line in your valid_file 
    # we will evaluate the performance of model on valid_file every epoch
    end_time = time.time()
    cost_time = end_time - start_time
    print('Time cost for training is {0:.2f} mins'.format((cost_time)/60.0))

    ckpt_path = tf.train.latest_checkpoint(model_path)
    model.load_model(ckpt_path)
   
    if flags_obj.write_prediction_to_file:
        res = model.run_weighted_eval_write(test_file, output_file,num_ngs=test_num_ngs)
    else:
        res = model.run_weighted_eval(test_file, num_ngs=test_num_ngs)

    print(flags_obj.name)
    print(res)
    
    


if __name__ == "__main__":

    app.run(main)
