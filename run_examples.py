import numpy as np
import tensorflow as tf
import deepcrispr as dc

## On-target Seq-only Regression Task
file_path = 'examples/eg_reg_on_target_seq.rsgt'
input_data = dc.Sgt(file_path, with_y=True)
x, y = input_data.get_dataset()
x = np.expand_dims(x, axis=2)  # shape(x) = [10, 4, 1, 23]
sess = tf.InteractiveSession()
on_target_model_dir = 'trained_models/ontar_cnn_reg_seq'
dcmodel = dc.DCModelOntar(sess, on_target_model_dir, is_reg=True, seq_feature_only=True)
predicted_on_target = dcmodel.ontar_predict(x)

## On-target Classification Task
file_path = 'examples/eg_cls_on_target.episgt'
input_data = dc.Episgt(file_path, num_epi_features=4, with_y=True)
x, y = input_data.get_dataset()
x = np.expand_dims(x, axis=2)  # shape(x) = [100, 8, 1, 23]
sess = tf.InteractiveSession()
on_target_model_dir = 'trained_models/ontar_ptaug_cnn'
dcmodel = dc.DCModelOntar(sess, on_target_model_dir, is_reg=False, seq_feature_only=False)
predicted_on_target = dcmodel.ontar_predict(x)


## On-target Regression Task
file_path = 'examples/eg_reg_on_target.repisgt'
input_data = dc.Episgt(file_path, num_epi_features=4, with_y=True)
x, y = input_data.get_dataset()
x = np.expand_dims(x, axis=2)  # shape(x) = [100, 8, 1, 23]
sess = tf.InteractiveSession()
on_target_model_dir = 'trained_models/ontar_pt_cnn_reg'
dcmodel = dc.DCModelOntar(sess, on_target_model_dir, is_reg=True, seq_feature_only=False)
predicted_on_target = dcmodel.ontar_predict(x)


## Off-target Classification Task
file_path = 'examples/eg_cls_off_target.epiotrt'
input_data = dc.Epiotrt(file_path, num_epi_features=4, with_y=True)
(x_on, x_off), y = input_data.get_dataset()
x_on = np.expand_dims(x_on, axis=2)  # shape(x) = [100, 8, 1, 23]
x_off = np.expand_dims(x_off, axis=2)  # shape(x) = [100, 8, 1, 23]
sess = tf.InteractiveSession()
off_target_model_dir = 'trained_models/offtar_pt_cnn'
dcmodel = dc.DCModelOfftar(sess, off_target_model_dir, is_reg=False)
predicted_off_target = dcmodel.offtar_predict(x_on, x_off)


## Off-target Regression Task
file_path = 'examples/eg_reg_off_target.repiotrt'
input_data = dc.Epiotrt(file_path, num_epi_features=4, with_y=True)
(x_on, x_off), y = input_data.get_dataset()
x_on = np.expand_dims(x_on, axis=2)  # shape(x) = [100, 8, 1, 23]
x_off = np.expand_dims(x_off, axis=2)  # shape(x) = [100, 8, 1, 23]
sess = tf.InteractiveSession()
off_target_model_dir = 'trained_models/offtar_pt_cnn_reg'
dcmodel = dc.DCModelOfftar(sess, off_target_model_dir, is_reg=True)
predicted_off_target = dcmodel.offtar_predict(x_on, x_off)