from __future__ import print_function
import os
from itertools import islice, chain
import numpy as np
np.random.seed(123)  # for reproducibility
import os
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, TimeDistributed, Masking
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
import random
random.seed(4)
from random import shuffle
import binary_io
#import print_weights2
import h5py
import sys

def load_datasets(data_path, filelist, feature_dimension, ext):
    startfile=1
    filenames=file(filelist).readlines()
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
        if startfile:
            startfile=0
            data_mat=load_mat
        else:
            data_mat=np.r_[data_mat, load_mat]
    return data_mat

def load_seq_datasets(data_path, filelist, timesteps, feature_dimension, ext, mask_value=0): #This pads zeros according to max seq len
    startfile=1
    num_records=0
    filenames=file(filelist).readlines()
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
        remaining_frames= load_mat.shape[0] % timesteps
        
        num_mask_frames=0 if remaining_frames == 0 else timesteps - remaining_frames
        mask_mat =np.ones((num_mask_frames, load_mat.shape[1]), dtype=float) * float(mask_value)   

        if startfile:
            startfile=0
            data_mat=np.r_[load_mat, mask_mat]
            num_records += np.ceil(load_mat.shape[0] / float(timesteps))
        else:
            data_mat=np.r_[data_mat, load_mat, mask_mat]
            num_records += np.ceil(load_mat.shape[0] / float(timesteps))
    #num_records=num_of expected time chunks for BPTT
    #The input to LSTM needs to be a 3-D tensor of shape: num_of_utterances x num_timesteps x feature_dimension
    reshaped_data_mat=np.reshape(data_mat, (-1, timesteps, feature_dimension))
    assert  (reshaped_data_mat.shape[0] == num_records), "Num of records do not match reshpaed matrix. num of records = %d, reshaped size = %d" % (num_records, reshaped_data_mat.shape[0]) 
    return reshaped_data_mat

def load_test_seq(data_path, filename, timesteps, feature_dimension, ext, mask_value=0): #This pads zeros according to max seq len
    merlin_io=binary_io.BinaryIOCollection()
    load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
    true_length=load_mat.shape[0]
    remaining_frames= load_mat.shape[0] % timesteps
    num_mask_frames=0 if remaining_frames == 0 else timesteps - remaining_frames
    mask_mat =np.ones((num_mask_frames, load_mat.shape[1]), dtype=float) * float(mask_value)   
    data_mat=np.r_[load_mat, mask_mat]
    reshaped_data_mat=np.reshape(data_mat, (-1, timesteps, feature_dimension))
    return reshaped_data_mat, true_length

def load_seq_datasets(data_path, filelist, timesteps, feature_dimension, ext, mask_value=0): #This pads zeros according to max seq len
    startfile=1
    num_records=0
    filenames=file(filelist).readlines()
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
        remaining_frames= load_mat.shape[0] % timesteps
        
        num_mask_frames=0 if remaining_frames == 0 else timesteps - remaining_frames
        mask_mat =np.ones((num_mask_frames, load_mat.shape[1]), dtype=float) * float(mask_value)   

        if startfile:
            startfile=0
            data_mat=np.r_[load_mat, mask_mat]
            num_records += np.ceil(load_mat.shape[0] / float(timesteps))
        else:
            data_mat=np.r_[data_mat, load_mat, mask_mat]
            num_records += np.ceil(load_mat.shape[0] / float(timesteps))
    #The input to LSTM needs to be a 3-D tensor of shape: num_of_utterances x num_timesteps x feature_dimension
    reshaped_data_mat=np.reshape(data_mat, (-1, timesteps, feature_dimension))
    assert  (reshaped_data_mat.shape[0] == num_records), "Num of records do not match reshpaed matrix. num of records = %d, reshaped size = %d" % (num_records, reshaped_data_mat.shape[0]) 
    return reshaped_data_mat


def binary_file_length(filename, feature_dimension):
    merlin_io=binary_io.BinaryIOCollection()
    load_mat=merlin_io.load_bianry_file(filename, feature_dimension)
    return  int(load_mat.shape[0])

def get_max_seq_len(datapath, filelist, feature_dimension, ext):
    max_seq_len =int(0)
    for filename in filelist:
        flength=binary_file_length(datapath + filename.strip() + ext, feature_dimension)
        max_seq_len=max_seq_len if max_seq_len >flength else flength
    return max_seq_len

def load_batch_datasets(data_path, filenames, feature_dimension, ext):
    startfile=1
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
        if startfile:
            startfile=0
            data_mat=load_mat
        else:
            data_mat=np.r_[data_mat, load_mat]
    return data_mat

def load_seq_batch_datasets(data_path, filenames, timesteps, feature_dimension, ext, mask_value=0): #This pads zeros according to max seq len
    startfile=1
    num_records=0
    merlin_io=binary_io.BinaryIOCollection()
    for filename in filenames:
        load_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
        remaining_frames= load_mat.shape[0] % timesteps
        
        num_mask_frames=0 if remaining_frames == 0 else timesteps - remaining_frames
        mask_mat =np.ones((num_mask_frames, load_mat.shape[1]), dtype=float) * float(mask_value)   
        if startfile:
            startfile=0
            data_mat=np.r_[load_mat, mask_mat]
            num_records += np.ceil(load_mat.shape[0] / float(timesteps))
        else:
            data_mat=np.r_[data_mat, load_mat, mask_mat]
            num_records += np.ceil(load_mat.shape[0] / float(timesteps))
    #The input to LSTM needs to be a 3-D tensor of shape: num_of_utterances x num_timesteps x feature_dimension
    reshaped_data_mat=np.reshape(data_mat, (-1, timesteps, feature_dimension))
    assert  (reshaped_data_mat.shape[0] == num_records), "Num of records do not match reshpaed matrix. num of records = %d, reshaped size = %d" % (num_records, reshaped_data_mat.shape[0]) 
    return reshaped_data_mat


def batchgen(iterable, size):#This  takes the list of chuked filenames and returns the conatenated matrix of those files.
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        #yield batchiter
        yield chain([batchiter.next()], batchiter)

def shuffle_filelist(filelist, output_file='random_list'): # To make the filelist shuffling deterministic on random seed
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

def build_model(n_input_dim=None, n_output_dim=None, timesteps=None, hidden_size=128, stateful=False, mask_value=0):
    model = Sequential()
   # model.add(Masking(mask_value=mask_value, input_shape=(None, n_input_dim)))
    model.add(LSTM(hidden_size, return_sequences=True,stateful=stateful,
              input_shape=(timesteps, n_input_dim),
              W_regularizer=l2(0.00001), U_regularizer=l2(0.0001))) 
    model.add(LSTM(hidden_size, return_sequences=True, stateful=stateful,
                        input_shape=(timesteps, hidden_size),
                        W_regularizer=l2(0.00001), U_regularizer=l2(0.0001)))
    model.add(LSTM(hidden_size, return_sequences=True, stateful=stateful,
                                      input_shape=(timesteps, hidden_size),
                                      W_regularizer=l2(0.00001), U_regularizer=l2(0.0001)))  
    model.add(LSTM(hidden_size, return_sequences=True, stateful=stateful,
                                      input_shape=(timesteps, hidden_size),
                                      W_regularizer=l2(0.00001), U_regularizer=l2(0.0001))) 
    model.add(LSTM(hidden_size, return_sequences=True, stateful=stateful,
                                      input_shape=(timesteps, hidden_size),
                                      W_regularizer=l2(0.00001), U_regularizer=l2(0.0001))) 

    model.add(LSTM(n_output_dim, input_shape=(timesteps, hidden_size), return_sequences=True, 
            stateful=stateful, W_regularizer=l2(0.00001), U_regularizer=l2(0.0001)))
    model.add(TimeDistributed(Dense(n_output_dim, activation='linear',input_shape=(timesteps, n_output_dim),W_regularizer=l2(0.00001))))

    model.summary()
    return model

def train_tts(datapath, batch_size=128, nb_epoch=20, chunksize=20, timesteps=300, feature_dimension=[711,199], ext=['.lab', '.cmp'], mask_value=0, use_sample_weights=False):
    train_list = datapath + 'train_list'
    test_list = datapath + 'test_list'
    val_list = datapath + 'val_list'
    model=build_model(n_input_dim=feature_dimension[0], n_output_dim=feature_dimension[1], timesteps=timesteps, hidden_size=256,  stateful=False, mask_value=mask_value)
    sgd = SGD(lr=0.04, momentum =0.5, clipvalue=0.01, decay=1e-8)
    if use_sample_weights:
        model.compile(loss='mse', optimizer='adam', sample_weight_mode="temporal")
    else:
        model.compile(loss='mean_sum_error', optimizer='adam')
    X_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[1]), ext[1]) 
    X_test = load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_test= load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[1]), ext[1]) 
    
    for epoch_num in range(nb_epoch):
        print ("Epoch %d / %d" % (epoch_num + 1, nb_epoch) )
        #shuffle the list
        #os.system('cat %s|shuf > random_list' % val_list)
        shuffle_filelist(train_list)
        file_seq = open('random_list', 'r')
        for filenames_chunk in batchgen(file_seq, chunksize):
            print("Chunk :"),
            filelist=list(filenames_chunk)
            X_train = load_seq_batch_datasets(datapath, filelist, timesteps, feature_dimension[0], ext[0])
            Y_train = load_seq_batch_datasets(datapath, filelist, timesteps, feature_dimension[1], ext[1])
            print('Input Shape:', X_train.shape)
            print('Output Shape:', Y_train.shape)
            if use_sample_weights:
                sample_weights = np.where(Y_train == mask_value, 0., 1.) #This needs to be samples x timesteps, so reducing it along the feature dimension.
                sample_weight= np.amax(sample_weights,axis=2)
                model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=(X_val, Y_val), shuffle=True, sample_weight=sample_weight)
            
         #       val_score = model.evaluate(X_val, Y_val, verbose=2, sample_weight=sample_weight)
            else:
                model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=(X_val, Y_val), shuffle=True)
        file_seq.close() 
        #        val_score = model.evaluate(X_val, Y_val, verbose=2)
        if  use_sample_weights:
            val_sample_weights = np.where(Y_val == mask_value, 0., 1.) #This needs to be samples x timesteps, so reducing it along the feature dimension.
            val_sample_weight= np.amax(val_sample_weights,axis=2)
 
            val_score = model.evaluate(X_val, Y_val, verbose=2, sample_weight=val_sample_weight)
        else:
            val_score = model.evaluate(X_val, Y_val, verbose=2)
        print('Validation score for Epoch %d: %f' % (epoch_num +1, val_score))
        #os.system('rm random_list')--not required done in shuffle_filelist
    if use_sample_weights:
        test_sample_weights = np.where(Y_test == mask_value, 0., 1.) #This needs to be samples x timesteps, so reducing it along the feature dimension.
        test_sample_weight= np.amax(test_sample_weights,axis=2)
 
        test_score = model.evaluate(X_test, Y_test, verbose=2, sample_weight=test_sample_weight)
    else:
        test_score = model.evaluate(X_test, Y_test, verbose=2)
    print('Test score :', test_score)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("spasm_lstm_model_tts.json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("spasm_lstm_model_tts.h5", overwrite=True)
    print("Saved model to disk")
    return

def train_acoustic(Y_train, Y_val, Y_test):
    model=build_model(n_input_dim=199, n_output_dim=199)
    sgd = SGD(lr=0.00004, momentum =0.9, clipvalue=0.01, decay=1e-8)
    model.compile(loss='mean_sum_error', optimizer=sgd)
    model.fit(Y_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=2, validation_data=(Y_val, Y_val), shuffle=True)
    score = model.evaluate(Y_test, Y_test, verbose=2)
    print('Test score:', score)

    # serialize model to JSON
    model_json = model.to_json()
    with open("spasm_lstm_model_acoustic.json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("spasm_lstm_model_acoustic.h5")
    print("Saved model to disk")
    return

def train_linguistic(X_train, X_val, X_test):
    model=build_model(n_input_dim=711, n_output_dim=711)
    sgd = SGD(lr=0.0004, momentum =0.5, clipvalue=0.01, decay=1e-8)
    model.compile(loss='mean_sum_error', optimizer=sgd)
    model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=2, validation_data=(X_val, X_val), shuffle=True)
    score = model.evaluate(X_test, X_test, verbose=2)
    print('Test score:', score)

    # serialize model to JSON
    model_json = model.to_json()
    with open("spasm_lstm_model_linguistic.json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("spasm_lstm_model_linguistic.h5")
    print("Saved model to disk")
    return

def load_pretrain(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    model=build_model(n_input_dim=711, n_output_dim=199)
    sgd = SGD(lr=0.00004, momentum =0.5, clipvalue=0.5, decay=1e-8)
# load json and create model
    #json_file = open('spasm_lstm_modelL.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #linguistic_model = model_from_json(loaded_model_json)
    # load weights into new model
    #model.load_weights("spasm_lstm_modelL.h5")
    json_file = open('spasm_lstm_model_linguistic.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    linguistic_model = model_from_json(loaded_model_json)
    linguistic_model.load_weights("spasm_lstm_model_linguistic.h5")
    
    json_file = open('spasm_lstm_model_acoustic.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    acoustic_model = model_from_json(loaded_model_json)
    acoustic_model.load_weights("spasm_lstm_model_acoustic.h5")
    am =h5py.File("spasm_lstm_model_acoustic.h5") 
    #for k in range(2, am.attrs['nb_layers']):
    #    g = f['layer_{}'.format(k)]
    #    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #    model.layers[k].set_weights(weights)
    #am.close()
    #print('Model loaded.')
    for i in range(0, 5):
        model.layers[i].set_weights(linguistic_model.layers[i].get_weights())
    for i in range(5,len(acoustic_model.layers)):
        #weights=(print_weights2.get_weightsc('spasm_lstm_model_acoustic.h5',int(i)))
        model.layers[i].set_weights(acoustic_model.layers[i].get_weights())       # model.weights[i].set_weights(am.weights[i])
    #print("Loaded model from disk")
    model.compile(loss='mean_sum_error', optimizer=sgd)
    model.fit(X_train, Y_train, batch_size=batch_size, 
        nb_epoch=nb_epoch, verbose=2, validation_data=(X_val, Y_val),shuffle=True)
    # evaluate loaded model on test data
    score = model.evaluate(X_test, Y_test, verbose=2)
    print('Test score:', score)

    # serialize model to JSON
    model_json = model.to_json()
    with open("spasm_lstm_model_pretrain.json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("spasm_lstm_model_pretrain.h5")
    print("Saved model to disk")
    return

def retrain(X_train, Y_train, X_val, Y_val, X_test, Y_test, model_name, nb_epoch=20, batch_size=128):
    #model_name=sys.argv[2]
    sgd = SGD(lr=0.0004, decay=1e-8,momentum=0.9, clipvalue=0.01)

# load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")
    print("Loaded model from disk")

    loaded_model.compile(loss='mean_sum_error', optimizer=sgd)

    loaded_model.fit(X_train, Y_train, batch_size=batch_size, 
        nb_epoch=nb_epoch, verbose=2, validation_data=(X_val, Y_val),shuffle=True)
    # evaluate loaded model on test data
    score = loaded_model.evaluate(X_test, Y_test, verbose=2)
    print('Test score:', score)

    # serialize model to JSON
    model_json = loaded_model.to_json()
    with open(model_name+'.json', "w") as json_file:
            json_file.write(loaded_model_json)
    # serialize weights to HDF5
    loaded_model.save_weights(model_name+'.h5')
    print("Saved model to disk")
    return


def predict_merlin(data_path, test_filelist, model_name, save_dir, timesteps=300,n_input_dim=711, n_output_dim=199):
    sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=False, clipnorm=0.01)
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+'.h5')
    print("Loaded model from disk")
    loaded_model.compile(loss='mean_sum_error', optimizer=sgd)
    merlin_io = binary_io.BinaryIOCollection() 
    with open(test_filelist, 'r') as fl:
        for filename in fl:
            print(filename)
            test_x, test_seq_length = load_test_seq(data_path, filename, timesteps=timesteps, feature_dimension=n_input_dim, ext='.lab', mask_value=0)
            prediction = loaded_model.predict(test_x, batch_size=1, 
                        verbose=2)
            actual_prediction=np.reshape(prediction, (-1, n_output_dim))
            final_prediction=actual_prediction[:test_seq_length]
            if not os.path.exists( save_dir):
                 os.makedirs(save_dir)
            fname=  save_dir + filename.strip() + '.cmp'
            merlin_io.array_to_binary_file(final_prediction, fname)
    print("predicted features saved in  ", save_dir)
    return

def predict_merlin_acoustic(test_filelist):
    test_filelist = sys.argv[2]
    sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=False, clipnorm=0.01)
    json_file = open('spasm_lstm_model_acoustic.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("spasm_lstm_model_acoustic.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss='mse', optimizer=sgd)
    merlin_io = binary_io.BinaryIOCollection()
    with open(test_filelist, 'r') as fl:
        for filename in fl:
            print(filename)
            lfname = '/home2/pbaljeka/new_exp4/data_slt_full/data/test_mcep/' +filename.strip() +'.cmp'
            test_x=load_seq_datasets(data_path, filelist, timesteps, feature_dimension, ext, mask_value=0)
            #test_x = merlin_io.load_binary_file(lfname, 199)
            prediction = loaded_model.predict(test_x, batch_size=1, 
                        verbose=2)
            if not os.path.exists('/home2/pbaljeka/new_exp4/data_slt_full/predicted_features/merlin_mlp_acoustic/'):
                 os.makedirs('/home2/pbaljeka/new_exp4/data_slt_full/predicted_features/merlin_mlp_acoustic/')
            fname='/home2/pbaljeka/new_exp4/data_slt_full/predicted_features/merlin_mlp_acoustic/' + filename.strip() + '.cmp'
            merlin_io.array_to_binary_file(prediction, fname)
    return


if __name__=='__main__':
    option=sys.argv[1]
    data_path='/home2/pbaljeka/new_exp4/data_slt_full/data/'
    #data_path='/home2/pbaljeka/english_spasm_experiments/Data/all_data/' 
    train_list=data_path + 'train_list'
    val_list=data_path + 'val_list' 
    test_list=data_path + 'chk_list'
    feature_dimension=[711, 199]
    ext=['.lab', '.cmp']
    batch_size = 128
    nb_epoch = 20
    save_dir='/home2/pbaljeka/new_exp4/data_slt_full/predicted_features/merlin_lstm_tts_256-nomask/'
    #save_dir='/home2/pbaljeka/english_spasm_experiments/Data/predicted_features/merlin_lstm_tts_v2_128/'
    timesteps=300

  

    if option == 'train_tts':
        train_tts(data_path)
    elif option == 'check':
        shuffle_filelist(test_list)
    elif option ==  'train_linguistic':
        pass
    elif option == "predict_merlin":
        predict_merlin(data_path=data_path, test_filelist=test_list,model_name='merlin_lstm_model_tts', save_dir=save_dir, timesteps=300)

