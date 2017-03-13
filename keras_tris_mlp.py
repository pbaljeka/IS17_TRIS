from __future__ import print_function
import os
from itertools import islice, chain
import numpy as np
np.random.seed(123)  # for reproducibility
import os
from keras.datasets import mnist
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Merge, merge
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
import binary_io
import random
random.seed(4)
from random import shuffle
#import print_weights2
import h5py
import sys

def load_stats(data_path, filelist, feature_dimension, ext):
    startfile=1
    filenames=file(filelist).readlines()
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        ym=merlin_io.load_binary_file(data_path + filename.strip() + '.ymean', feature_dimension)
        ys=merlin_io.load_binary_file(data_path + filename.strip() + '.ystd', feature_dimension)
        nm=merlin_io.load_binary_file(data_path + filename.strip() + '.nmean', feature_dimension)
        ns=merlin_io.load_binary_file(data_path + filename.strip() + '.nstd', feature_dimension)
        if startfile:
            startfile=0
            ymean=ym
            ystd=ys
            nmean=nm
            nstd=ns
        else:
            ymean=np.r_[ymean, ym]
            ystd=np.r_[ystd, ys]
            nmean=np.r_[nmean, nm]
            nstd=np.r_[nstd, ns]
    return ymean,ystd,nmean,nstd

def load_batch_stats(data_path, filenames, feature_dimension, ext):
    startfile=1
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        ym=merlin_io.load_binary_file(data_path + filename.strip() + '.ymean', feature_dimension)
        ys=merlin_io.load_binary_file(data_path + filename.strip() + '.ystd', feature_dimension)
        nm=merlin_io.load_binary_file(data_path + filename.strip() + '.nmean', feature_dimension)
        ns=merlin_io.load_binary_file(data_path + filename.strip() + '.nstd', feature_dimension)
        if startfile:
            startfile=0
            ymean=ym
            ystd=ys
            nmean=nm
            nstd=ns
        else:
            ymean=np.r_[ymean, ym]
            ystd=np.r_[ystd, ys]
            nmean=np.r_[nmean, nm]
            nstd=np.r_[nstd, ns]
    return ymean,ystd,nmean,nstd

def load_tris_data_file(data_path, predict_data_path, filename, feature_dimension, ext):
    merlin_io=binary_io.BinaryIOCollection()
    load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
    mean_feat=merlin_io.load_binary_file(predict_data_path + filename.strip() + '.mean', 52)
    std_feat=merlin_io.load_binary_file(predict_data_path + filename.strip() + '.std', 52)
    qa = load_mat[:,:777]
    assert qa.shape[1] == 777
    ling_feat=load_mat[:,777:]
    assert ling_feat.shape[1] == 711
    return ling_feat, qa, mean_feat, std_feat

def load_tris_data(data_path, filelist, feature_dimension, ext):
    startfile=1
    filenames=file(filelist).readlines()
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
        mean_feat=merlin_io.load_binary_file(data_path + filename.strip() + '.mean', 52)
        std_feat=merlin_io.load_binary_file(data_path + filename.strip() + '.std', 52)
        qa = load_mat[:,:777]
        assert qa.shape[1] == 777
        ling_feat=load_mat[:,777:]
        assert ling_feat.shape[1] == 711

        if startfile:
            startfile=0
            ling_mat=ling_feat
            qa_mat = qa
            mean_mat = mean_feat
            std_mat = std_feat
        else:
            ling_mat=np.r_[ling_mat, ling_feat]
            qa_mat=np.r_[qa_mat, qa]
            mean_mat=np.r_[mean_mat, mean_feat]
            std_mat = np.r_[std_mat, std_feat]
    return ling_mat, qa_mat, mean_mat, std_mat

def load_batch_tris_data(data_path, filenames, feature_dimension, ext):
    startfile=1
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
        mean_feat=merlin_io.load_binary_file(data_path + filename.strip() + '.mean', 52)
        std_feat=merlin_io.load_binary_file(data_path + filename.strip() + '.std', 52)
        qa = load_mat[:, :777]
        assert qa.shape[1] == 777
        ling_feat=load_mat[:, 777:]
        assert ling_feat.shape[1] == 711

        if startfile:
            startfile=0
            ling_mat=ling_feat
            qa_mat = qa
            mean_mat = mean_feat
            std_mat = std_feat
        else:
            ling_mat=np.r_[ling_mat, ling_feat]
            qa_mat=np.r_[qa_mat, qa]
            mean_mat=np.r_[mean_mat, mean_feat]
            std_mat = np.r_[std_mat, std_feat]
    return ling_mat, qa_mat, mean_mat, std_mat

def batchgen(iterable, size):#This  takes the list of chuked frames and retuns a chunk
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([batchiter.next()], batchiter)

def shuffle_filelist(filelist, output_file='mlp_random_list'): # To make the filelist shuffling deterministic on random seed
    try:
        os.remove(output_file)
    except OSError:
        pass
    list_of_files=[]
    with open(filelist, 'r') as f:
        for filename in f:
            list_of_files.append(filename.strip())
    shuffle(list_of_files)
    with open(output_file, 'w') as f:
        for filename in list_of_files:
            f.write(filename + '\n')
    return


def build_func_model(hidden_size=64):
    ling_in = Input(shape=(711,), name='ling_in')
    ling_emb = Dense(128, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(ling_in)
    ling_emb = Dense(64, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='ling_emb')(ling_emb)
    ling_embD = Dense(128, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(ling_emb)
    ling_out = Dense(711, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='ling_out')(ling_embD)

    QA_in = Input(shape=(777,), name='QA_in')
    QA_emb = Dense(128, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(QA_in)
    QA_emb = Dense(70, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='QA_emb')(QA_emb)
    QA_embD = Dense(128, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(QA_emb)
    QA_out = Dense(777, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='QA_out')(QA_embD)

    AD_mean = Input(shape=(52,), name='AD_mean')
    AD_std = Input(shape=(52,), name='AD_std')
    x= merge([ling_emb, QA_emb, AD_mean, AD_std], mode='concat')
    x = Dense(hidden_size, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(x)
    x = Dense(hidden_size, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(x)
    x = Dense(hidden_size, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(x)
    x = Dense(hidden_size, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001))(x)

    ymean = Dense(52, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='ymean')(x)
    ystd = Dense(52, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='ystd')(x)
    nmean = Dense(52, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='nmean')(x)
    nstd = Dense(52, init='glorot_normal', activation='tanh', W_regularizer=l2(0.000001), name='nstd')(x)
    model=Model(input=[ling_in, QA_in, AD_mean, AD_std], output=[ymean, ystd, nmean, nstd, ling_out, QA_out])
    model.summary()
    return model

def best_checkpoint_model(delta_metric, model, best_model_filepath, patience_counter, early_stop=True, patience=3, delta_thresh=0.05):# This does early stopping and saves only the best model.
    end_training=False
    if delta_metric > delta_thresh:#Save the model
        model_json = model.to_json()
        with open(model_filepath + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
            model.save_weights(model_filepath + ".h5", overwrite=True)
        print("Saved best model to disk")
        patience_counter=0 #reset
    else:
        patience_counter +=1
    if patience_counter==patience:
        end_training = True
    return end_training, patience_counter

def checkpoint_model(delta_metric, model, model_filepath, patience_counter, early_stop=True, patience=3, delta_thresh=0.05):# This does early stopping and saves the model after each epoch.
    end_training=False
    model_json = model.to_json()
    with open(model_filepath + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_filepath + ".h5", overwrite=False)
    print("Saved saved model to disk")

    if delta_metric > delta_thresh:
        patience_counter = 0
    else:
        patience_counter +=1
    if patience_counter==patience:
        end_training = True
    return end_training, patience_counter

def save_model(model, model_filepath):# This does early stopping and saves the model after each epoch.
    model_json = model.to_json()
    with open(model_filepath + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_filepath + ".h5", overwrite=True)
    print("Saved saved model to disk")

def train_tris(model_name, batch_size=256, nb_epoch=20, chunksize=20, feature_dimension=[1488, 52], ext=['.tris', '.ymean', '.ystd', '.nmean', '.nstd']):
    utils_path='/home/pbaljeka/TRIS_Exps3/utils/'
    data_path='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/norm_nn_tris/'
    train_list = utils_path + 'all_list'
    test_list = utils_path + 'test_list'
    val_list = utils_path + 'val_list'
    random_filelist = model_name + '_random_list'
    model=build_func_model(hidden_size=64)
    sgd = SGD(lr=0.04, momentum =0.5, clipvalue=0.01, decay=1e-8)
    model.compile(loss={'ymean': 'mse','ystd': 'mse', 'nmean': 'mse','nstd': 'mse', 'ling_out': 'mse','QA_out': 'mse'}, loss_weights={'ymean':1., 'ystd':.6, 'nmean':1., 'nstd':0.6, 'ling_out':0.1, 'QA_out':0.1},  optimizer='adam')
    ymean_val, ystd_val, nmean_val, nstd_val = load_stats(data_path, val_list, int(feature_dimension[1]), ext)
    ling_val, qa_val, mean_val, std_val = load_tris_data(data_path, val_list, int(feature_dimension[0]), ext[0])

    # checkpoint
    #prev_val_score=np.zeros(3)
    #patience_counter=0
    model_dir= utils_path + '/models/' + model_name +'/'
    logfile= utils_path + model_name + '_logfile.log'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch_num in range(nb_epoch):
        print ("Epoch %d / %d" % (epoch_num + 1, nb_epoch) )
        #shuffle the list

        model_filepath=model_dir +'/chkpt-' + str(epoch_num + 1)
        #os.system('cat %s|shuf > %s' % (train_list, random_filelist))
        shuffle_filelist(train_list, random_filelist)
        file_seq = open(random_filelist, 'r')
        for filenames_chunk in batchgen(file_seq, chunksize):
            print("Chunk :"),
            filelist=list(filenames_chunk)
            ymean_train, ystd_train, nmean_train, nstd_train = load_batch_stats(data_path, filelist, int(feature_dimension[1]), ext)
            ling_train, qa_train, mean_train, std_train = load_batch_tris_data(data_path, filelist, int(feature_dimension[0]), ext[0])

            print('Ling Shape:', ling_train.shape)
            print('QA Shape:', qa_train.shape)
            print('mean Shape:', mean_train.shape)
            print('std Shape:', std_train.shape)
            model.fit({'ling_in':ling_train, 'QA_in':qa_train, 'AD_mean':mean_train, 'AD_std':std_train}, {'ymean':ymean_train, 'ystd':ystd_train, 'nmean':nmean_train, 'nstd':nstd_train, 'ling_out':ling_train, 'QA_out':qa_train}, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=({'ling_in':ling_val, 'QA_in':qa_val, 'AD_mean':mean_val, 'AD_std':std_val}, {'ymean':ymean_val, 'ystd':ystd_val, 'nmean':nmean_val, 'nstd':nstd_val, 'ling_out':ling_val, 'QA_out':qa_val}), shuffle=True)
        file_seq.close()
        val_score = model.evaluate({'ling_in':ling_val, 'QA_in':qa_val, 'AD_mean': mean_val, 'AD_std': std_val}, {'ymean':ymean_val, 'ystd':ystd_val, 'nmean':nmean_val, 'nstd':nstd_val, 'ling_out':ling_val, 'QA_out':qa_val},verbose=2)
        save_model(model, model_filepath)
        with open(logfile, 'a+') as logf:
            logf.write(' Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[0]))
            logf.write('Y_MEAN: Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[1]))
            logf.write('Y_STD: Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[2]))
            logf.write('N_MEAN: Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[3]))
            logf.write('N_STD:Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[4]))

    ling_test, qa_test, mean_test, std_test = load_tris_data(data_path, test_list, int(feature_dimension[0]), ext[0])
    ymean_test, ystd_test, nmean_test, nstd_test = load_stats(data_path, test_list, int(feature_dimension[1]), ext)
    test_score = model.evaluate({'ling_in':ling_test, 'QA_in':qa_test, 'AD_mean': mean_test, 'AD_std': std_test}, {'ymean':ymean_test, 'ystd':ystd_test, 'nmean':nmean_test, 'nstd':nstd_test, 'ling_out':ling_test, 'QA_out':qa_test},verbose=2)
    with open(logfile, 'a+') as logf:
        logf.write('Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[0]))
        logf.write('Y_MEAN: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[1]))
        logf.write('Y_STD: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[2]))
        logf.write('N_MEAN: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[3]))
        logf.write('N_STD: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[4]))

    return

def adapt_tris(model_name, chkpt, batch_size=256, nb_epoch=20, chunksize=20, feature_dimension=[1488, 52], ext=['.tris', '.ymean', '.ystd', '.nmean', '.nstd']):
    utils_path='/home/pbaljeka/TRIS_Exps2/cmu_us_slt-tris_utils/'
    data_path='/home/pbaljeka/TRIS_Exps2/clb_exps/cmu_us_clb5fullehmm/festival/norm_nn_tris/'
    train_list = utils_path + 'all_list'
    test_list = utils_path + 'test_list'
    val_list = utils_path + 'val_list'
    random_filelist = model_name + 'adapt_random_list'
    ymean_val, ystd_val, nmean_val, nstd_val = load_stats(data_path, val_list, int(feature_dimension[1]), ext)
    ling_val, qa_val, mean_val, std_val = load_tris_data(data_path, val_list, int(feature_dimension[0]), ext[0])

    # checkpoint
    #prev_val_score=np.zeros(3)
    #patience_counter=0
    model_dir= utils_path + '/models/' + model_name +'/'
    model_name_c = model_dir  +'/chkpt-'+str(chkpt)
    logfile= utils_path + model_name + '_adapt_logfile.log'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    json_file = open(model_name_c + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_name_c+".h5")
    print("Loaded model from disk")

    model.compile(loss={'ymean': 'mse','ystd': 'mse', 'nmean': 'mse','nstd': 'mse', 'ling_out': 'mse','QA_out': 'mse'}, loss_weights={'ymean':1., 'ystd':1., 'nmean':1., 'nstd':1., 'ling_out':0.1, 'QA_out':0.1},  optimizer='adam')

    for epoch_num in range(nb_epoch):
        print ("Epoch %d / %d" % (epoch_num + 1, nb_epoch) )
        #shuffle the list

        model_filepath=model_dir +'/adapt-chkpt-' + str(epoch_num + 1)
        #os.system('cat %s|shuf > %s' % (train_list, random_filelist))
        shuffle_filelist(train_list, random_filelist)
        file_seq = open(random_filelist, 'r')
        for filenames_chunk in batchgen(file_seq, chunksize):
            print("Chunk :"),
            filelist=list(filenames_chunk)
            ymean_train, ystd_train, nmean_train, nstd_train = load_batch_stats(data_path, filelist, int(feature_dimension[1]), ext)
            ling_train, qa_train, mean_train, std_train = load_batch_tris_data(data_path, filelist, int(feature_dimension[0]), ext[0])

            print('Ling Shape:', ling_train.shape)
            print('QA Shape:', qa_train.shape)
            print('mean Shape:', mean_train.shape)
            print('std Shape:', std_train.shape)
            model.fit({'ling_in':ling_train, 'QA_in':qa_train, 'AD_mean':mean_train, 'AD_std':std_train}, {'ymean':ymean_train, 'ystd':ystd_train, 'nmean':nmean_train, 'nstd':nstd_train, 'ling_out':ling_train, 'QA_out':qa_train}, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=({'ling_in':ling_val, 'QA_in':qa_val, 'AD_mean':mean_val, 'AD_std':std_val}, {'ymean':ymean_val, 'ystd':ystd_val, 'nmean':nmean_val, 'nstd':nstd_val, 'ling_out':ling_val, 'QA_out':qa_val}), shuffle=True)
        file_seq.close()
        val_score = model.evaluate({'ling_in':ling_val, 'QA_in':qa_val, 'AD_mean': mean_val, 'AD_std': std_val}, {'ymean':ymean_val, 'ystd':ystd_val, 'nmean':nmean_val, 'nstd':nstd_val, 'ling_out':ling_val, 'QA_out':qa_val},verbose=2)
        save_model(model, model_filepath)
        with open(logfile, 'a+') as logf:
            logf.write('Y_MEAN: Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[0]))
            logf.write('Y_STD: Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[1]))
            logf.write('N_MEAN: Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[2]))
            logf.write('N_STD:Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score[3]))

    ling_test, qa_test, mean_test, std_test = load_tris_data(data_path, test_list, int(feature_dimension[0]), ext[0])
    ymean_test, ystd_test, nmean_test, nstd_test = load_stats(data_path, test_list, int(feature_dimension[1]), ext)
    test_score = model.evaluate({'ling_in':ling_test, 'QA_in':qa_test, 'AD_mean': mean_test, 'AD_std': std_test}, {'ymean':ymean_test, 'ystd':ystd_test, 'nmean':nmean_test, 'nstd':nstd_test, 'ling_out':ling_test, 'QA_out':qa_test},verbose=2)
    with open(logfile, 'a+') as logf:
        logf.write('Y_MEAN: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[0]))
        logf.write('Y_STD: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[1]))
        logf.write('N_MEAN: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[2]))
        logf.write('N_STD: Test score for Epoch %d: %f \n' % (epoch_num +1, test_score[3]))

    return

def retrain_tris(model_name, chkpt, batch_size=256, nb_epoch=20, chunksize=20, feature_dimension=[1488, 52], ext=['.tris', '.ymean', '.ystd', '.nmean', '.nstd']):
    utils_path='/home/pbaljeka/TRIS_Exps2/cmu_us_slt-tris_utils/'
    data_path='/home/pbaljeka/TRIS_Exps2/cmu_us_slt/festival/norm_nn_tris/'
    train_list = utils_path + 'all_list'
    test_list = utils_path + 'test_list'
    val_list = utils_path + 'val_list'
    random_filelist = model_name + '_random_list'
    ymean_val, ystd_val, nmean_val, nstd_val = load_stats(data_path, val_list, int(feature_dimension[1]), ext)
    ling_val, qa_val, mean_val, std_val = load_tris_data(data_path, val_list, int(feature_dimension[0]), ext[0])

    # checkpoint
    #prev_val_score=np.zeros(3)
    #patience_counter=0
    model_dir= utils_path + '/models/' + model_name +'/'
    model_name_c= model_dir + '/chkpt-' +str(chkpt)
    logfile= utils_path + model_name + '_logfile.log'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    json_file = open(model_name_c + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_name_c+".h5")
    print("Loaded model from disk")

    model.compile(loss={'ymean': 'mse','ystd': 'mse', 'nmean': 'mse','nstd': 'mse', 'ling_out': 'mse','QA_out': 'mse'}, loss_weights={'ymean':1., 'ystd':1., 'nmean':1., 'nstd':1., 'ling_out':0.1, 'QA_out':0.1},  optimizer='adam')

    for epoch_num in range(nb_epoch):
        print ("Epoch %d / %d" % (epoch_num + 1, nb_epoch) )
        #shuffle the list

        model_filepath=model_dir +'/chkpt-' + str(epoch_num + int(chkpt) +1)
        #os.system('cat %s|shuf > %s' % (train_list, random_filelist))
        shuffle_filelist(train_list, random_filelist)
        file_seq = open(random_filelist, 'r')
        for filenames_chunk in batchgen(file_seq, chunksize):
            print("Chunk :"),
            filelist=list(filenames_chunk)
            ymean_train, ystd_train, nmean_train, nstd_train = load_batch_stats(data_path, filelist, int(feature_dimension[1]), ext)
            ling_train, qa_train, mean_train, std_train = load_batch_tris_data(data_path, filelist, int(feature_dimension[0]), ext[0])

            print('Ling Shape:', ling_train.shape)
            print('QA Shape:', qa_train.shape)
            print('mean Shape:', mean_train.shape)
            print('std Shape:', std_train.shape)
            model.fit({'ling_in':ling_train, 'QA_in':qa_train, 'AD_mean':mean_train, 'AD_std':std_train}, {'ymean':ymean_train, 'ystd':ystd_train, 'nmean':nmean_train, 'nstd':nstd_train, 'ling_out':ling_train, 'QA_out':qa_train}, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=({'ling_in':ling_val, 'QA_in':qa_val, 'AD_mean':mean_val, 'AD_std':std_val}, {'ymean':ymean_val, 'ystd':ystd_val, 'nmean':nmean_val, 'nstd':nstd_val, 'ling_out':ling_val, 'QA_out':qa_val}), shuffle=True)
        file_seq.close()
        val_score = model.evaluate({'ling_in':ling_val, 'QA_in':qa_val, 'AD_mean': mean_val, 'AD_std': std_val}, {'ymean':ymean_val, 'ystd':ystd_val, 'nmean':nmean_val, 'nstd':nstd_val, 'ling_out':ling_val, 'QA_out':qa_val},verbose=2)
        save_model(model, model_filepath)
        with open(logfile, 'a+') as logf:
            logf.write('Y_MEAN: Validation score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, val_score[0]))
            logf.write('Y_STD: Validation score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, val_score[1]))
            logf.write('N_MEAN: Validation score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, val_score[2]))
            logf.write('N_STD:Validation score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, val_score[3]))

    ling_test, qa_test, mean_test, std_test = load_tris_data(data_path, test_list, int(feature_dimension[0]), ext[0])
    ymean_test, ystd_test, nmean_test, nstd_test = load_stats(data_path, test_list, int(feature_dimension[1]), ext)
    test_score = model.evaluate({'ling_in':ling_test, 'QA_in':qa_test, 'AD_mean': mean_test, 'AD_std': std_test}, {'ymean':ymean_test, 'ystd':ystd_test, 'nmean':nmean_test, 'nstd':nstd_test, 'ling_out':ling_test, 'QA_out':qa_test},verbose=2)
    with open(logfile, 'a+') as logf:
        logf.write('Y_MEAN: Test score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, test_score[0]))
        logf.write('Y_STD: Test score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, test_score[1]))
        logf.write('N_MEAN: Test score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, test_score[2]))
        logf.write('N_STD: Test score for Epoch %d: %f \n' % (epoch_num +int(chkpt) +1, test_score[3]))

    return


def predict_tris(data_path, test_filelist, model_dir, chkpt,  save_dir,ext, n_input_dim=1488):
    #sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=False, clipnorm=0.01)
   # load weights into new model
    model_name= model_dir + 'chkpt-' + str(chkpt)
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(model_name+'.h5')
    print("Loaded model from disk")
    model.compile(loss={'ymean': 'mse','ystd': 'mse', 'nmean': 'mse','nstd': 'mse', 'ling_out': 'mse','QA_out': 'mse'}, loss_weights={'ymean':1., 'ystd':0.6, 'nmean':1., 'nstd':0.6, 'ling_out':0.1, 'QA_out':0.1},  optimizer='adam')

    merlin_io = binary_io.BinaryIOCollection()
    with open(test_filelist, 'r') as fl:
        for filename in fl:
            print(filename)
            ling_feat, qa, mean_feat, std_feat = load_tris_data_file(data_path, save_dir, filename, int(n_input_dim), ext=ext[0])
            prediction = model.predict({'ling_in':ling_feat, 'QA_in':qa, 'AD_mean':mean_feat, 'AD_std':std_feat}, batch_size=1,
                        verbose=2)
            print(len(prediction))
            if not os.path.exists( save_dir):
                 os.makedirs(save_dir)
            fname=  save_dir + filename.strip()
            merlin_io.array_to_binary_file(prediction[0], fname +'Y.mean')
            merlin_io.array_to_binary_file(prediction[1], fname + 'Y.std')
            merlin_io.array_to_binary_file(prediction[2], fname + 'N.mean')
            merlin_io.array_to_binary_file(prediction[3], fname + 'N.std')
    print("predicted features saved in  ", save_dir)
    return

def resynth():
    pass
if __name__=='__main__':
    option=sys.argv[1]
    feature_dimension=[1488, 52]
    ext=['.tris', '.ymean', '.ystd', '.nmean', '.nstd']
    batch_size = 64
    nb_epoch = 20
    chunksize=120
    chkpt=3
    model_name = 'slt_tris_norm_AD_1-FIX'
    test_list= '/home/pbaljeka/TRIS_Exps3/utils/nodenames'
    #save_dir='/home2/pbaljeka/english_spasm_experiments/Data/predicted_features/spasm_mlp_tts_v2/'
    save_dir='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/predicted_norm_nn_tris_trees/' +model_name + '/'
    data_path='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/norm_nn_tris_trees/'
    model_dir='/home/pbaljeka/TRIS_Exps3/utils/models/' + model_name + '/'
    if option == 'train':
        train_tris(model_name, int(batch_size), int(nb_epoch), int(chunksize), feature_dimension, ext)
    elif option == "predict":
        predict_tris(data_path, test_list,model_dir, chkpt, save_dir, ext)
    elif option == "retrain":
        retrain_tris(model_name, chkpt, int(batch_size), int(nb_epoch), int(chunksize), feature_dimension, ext)
    elif option == "resynth":
        pass

