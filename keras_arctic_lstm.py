from __future__ import print_function
import os
from itertools import islice, chain
import numpy as np
np.random.seed(123)  # for reproducibility
import os
from keras.datasets import mnist
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, merge, LSTM, TimeDistributed, Masking
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

def build_func_model(n_input_dim=None, n_output_dim=None, timesteps=None, hidden_size=128, stateful=False):
    linguistic_in = Input(shape=(timesteps, n_input_dim,), name='linguistic_input')
    x=LSTM(hidden_size, return_sequences=True, stateful=stateful, input_shape=(timesteps, hidden_size))(linguistic_in)
    x=LSTM(hidden_size, return_sequences=True, stateful=stateful, input_shape=(timesteps, hidden_size))(x)
    x=LSTM(hidden_size, return_sequences=True, stateful=stateful, input_shape=(timesteps, hidden_size))(x)
    x=LSTM(hidden_size, return_sequences=True, stateful=stateful, input_shape=(timesteps, hidden_size))(x)
    acoustic_out=TimeDistributed(Dense(n_output_dim, activation='linear', input_shape=(timesteps, hidden_size)))(x) 
    model=Model(input=linguistic_in, output=acoustic_out)
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

def train(main_path, model_name, batch_size=128, nb_epoch=20, timesteps=300, chunksize=20, feature_dimension=[711,199], ext=['.lab', '.cmp']):
    data_path= main_path + '/data/'
    train_list = data_path + 'train100_list'
    test_list = data_path + 'test100_list'
    val_list = data_path + 'val100_list'
    random_filelist = model_name + '_random_list'

    model=build_model(n_input_dim=feature_dimension[0], n_output_dim=feature_dimension[1], timesteps=timesteps, hidden_size=256,  stateful=False)
    #sgd = SGD(lr=0.04, momentum =0.5, clipvalue=0.01, decay=1e-8)
   
    model.compile(loss='mean_sum_error', optimizer='adam')
    X_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[1]), ext[1]) 
    X_test = load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_test= load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[1]), ext[1]) 
   # checkpoint
    prev_val_score=0
    patience_counter=0
    model_dir= main_path + '/models/' + model_name
    logfile=main_path + '/logfile_' + model_name + '.log'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch_num in range(nb_epoch):
        print ("Epoch %d / %d" % (epoch_num + 1, nb_epoch) )
        model_filepath=model_dir +'/chkpt-' + str(epoch_num + 1)
	#shuffle the list
        #os.system('cat %s|shuf > random_list' % val_list)
        shuffle_filelist(train_list, random_filelist)
        file_seq = open('random_list', 'r')
        for filenames_chunk in batchgen(file_seq, chunksize):
            print("Chunk :"),
            filelist=list(filenames_chunk)
            X_train = load_seq_batch_datasets(data_path, filelist, timesteps, feature_dimension[0], ext[0])
            Y_train = load_seq_batch_datasets(data_path, filelist, timesteps, feature_dimension[1], ext[1])
            print('Input Shape:', X_train.shape)
            print('Output Shape:', Y_train.shape)
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=(X_val, Y_val), shuffle=True)
        file_seq.close() 
        #        val_score = model.evaluate(X_val, Y_val, verbose=2)
        val_score = model.evaluate(X_val, Y_val, verbose=2)
        print('Validation score for Epoch %d: %f' % (epoch_num +1, val_score))
        #Checkpoint model for this epoch and check early stopping:
	delta_score=np.abs(val_score - prev_val_score)
        end_training, patience_counter=checkpoint_model(delta_score, model, model_filepath, patience_counter, early_stop=True, patience=3, delta_thresh=0.05)
        print('Validation score for Epoch %d: %f' % (epoch_num +1, val_score))
        with open(logfile, 'a+') as logf:
            logf.write('Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score))
        if end_training:
            print('Stopping early at epoch %d: '  % (epoch_num +1))
            with open(logfile, 'a+') as logf:
                logf.write('Stopping early for Epoch %d, with score : %f \n' % (epoch_num +1, val_score))
            break
        prev_val_score = val_score
    
    test_score = model.evaluate(X_test, Y_test, verbose=2)
    print('Test score :', test_score)
    
    with open(logfile, 'a+') as logf:
        logf.write('Test score is : %f \n' % (test_score))

    return

def train_func(main_path, model_name, batch_size=128, nb_epoch=20, timesteps=300, chunksize=20, feature_dimension=[711,199], ext=['.lab', '.cmp']):
    data_path= main_path + '/data/'
    train_list = data_path + 'train100_list'
    test_list = data_path + 'test100_list'
    val_list = data_path + 'val100_list'
    random_filelist = model_name + '_random_list'

    model=build_func_model(n_input_dim=feature_dimension[0], n_output_dim=feature_dimension[1], timesteps=timesteps, hidden_size=256,  stateful=False)
    #sgd = SGD(lr=0.04, momentum =0.5, clipvalue=0.01, decay=1e-8)
   
    model.compile(loss='mean_sum_error', optimizer='adam')
    X_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[1]), ext[1]) 
    X_test = load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_test= load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[1]), ext[1]) 
   # checkpoint
    prev_val_score=0
    patience_counter=0
    model_dir= main_path + '/models/' + model_name
    logfile=main_path + '/logfile_' + model_name + '.log'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch_num in range(nb_epoch):
        print ("Epoch %d / %d" % (epoch_num + 1, nb_epoch) )
        model_filepath=model_dir +'/chkpt-' + str(epoch_num + 1)
	#shuffle the list
        #os.system('cat %s|shuf > random_list' % val_list)
        shuffle_filelist(train_list, random_filelist)
        file_seq = open('random_list', 'r')
        for filenames_chunk in batchgen(file_seq, chunksize):
            print("Chunk :"),
            filelist=list(filenames_chunk)
            X_train = load_seq_batch_datasets(data_path, filelist, timesteps, feature_dimension[0], ext[0])
            Y_train = load_seq_batch_datasets(data_path, filelist, timesteps, feature_dimension[1], ext[1])
            print('Input Shape:', X_train.shape)
            print('Output Shape:', Y_train.shape)
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=(X_val, Y_val), shuffle=True)
        file_seq.close() 
        #        val_score = model.evaluate(X_val, Y_val, verbose=2)
        val_score = model.evaluate(X_val, Y_val, verbose=2)
        print('Validation score for Epoch %d: %f' % (epoch_num +1, val_score))
        #Checkpoint model for this epoch and check early stopping:
	delta_score=np.abs(val_score - prev_val_score)
        end_training, patience_counter=checkpoint_model(delta_score, model, model_filepath, patience_counter, early_stop=True, patience=3, delta_thresh=0.05)
        print('Validation score for Epoch %d: %f' % (epoch_num +1, val_score))
        with open(logfile, 'a+') as logf:
            logf.write('Validation score for Epoch %d: %f \n' % (epoch_num +1, val_score))
        if end_training:
            print('Stopping early at epoch %d: '  % (epoch_num +1))
            with open(logfile, 'a+') as logf:
                logf.write('Stopping early for Epoch %d, with score : %f \n' % (epoch_num +1, val_score))
            break
        prev_val_score = val_score
    
    test_score = model.evaluate(X_test, Y_test, verbose=2)
    print('Test score :', test_score)
    
    with open(logfile, 'a+') as logf:
        logf.write('Test score is : %f \n' % (test_score))

    return

def retrain(main_path, model_name, chkpt, batch_size=128, nb_epoch=20, timesteps=300, chunksize=20, feature_dimension=[711,199], ext=['.lab', '.cmp']):
    data_path= main_path + '/data/'
    train_list = data_path + 'train100_list'
    test_list = data_path + 'test100_list'
    val_list = data_path + 'val100_list'
    random_filelist = model_name + '_random_list'
    X_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_val = load_seq_datasets(data_path, val_list, timesteps, int(feature_dimension[1]), ext[1]) 
    X_test = load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[0]), ext[0]) 
    Y_test= load_seq_datasets(data_path, test_list, timesteps, int(feature_dimension[1]), ext[1]) 
   # checkpoint
    prev_val_score=0
    patience_counter=0
    model_dir= main_path + '/models/' + model_name
    logfile=main_path + '/logfile_' + model_name + '.log'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chkpt_model_name=model_dir +'/chkpt-' +str(chkpt)
    json_file = open(chkpt_model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(chkpt_model_name+'.h5')
    print("Loaded model from disk")

    loaded_model.compile(loss='mean_sum_error', optimizer='adam')

    for epoch_num in range(nb_epoch):
        print ("Epoch %d / %d" % (epoch_num + 1, nb_epoch) )
        model_filepath=model_dir +'/chkpt-' + str(epoch_num + int(chkpt)+ 1)
	#shuffle the list
        #os.system('cat %s|shuf > random_list' % val_list)
        shuffle_filelist(train_list, random_filelist)
        file_seq = open('random_list', 'r')
        for filenames_chunk in batchgen(file_seq, chunksize):
            print("Chunk :"),
            filelist=list(filenames_chunk)
            X_train = load_seq_batch_datasets(data_path, filelist, timesteps, feature_dimension[0], ext[0])
            Y_train = load_seq_batch_datasets(data_path, filelist, timesteps, feature_dimension[1], ext[1])
            print('Input Shape:', X_train.shape)
            print('Output Shape:', Y_train.shape)
            loaded_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=(X_val, Y_val), shuffle=True)
        file_seq.close() 
        #        val_score = model.evaluate(X_val, Y_val, verbose=2)
        val_score = loaded_model.evaluate(X_val, Y_val, verbose=2)
        print('Validation score for Epoch %d: %f' % (epoch_num +1, val_score))
        #Checkpoint model for this epoch and check early stopping:
	delta_score=np.abs(val_score - prev_val_score)
        end_training, patience_counter=checkpoint_model(delta_score, loaded_model, model_filepath, patience_counter, early_stop=True, patience=3, delta_thresh=0.05)
        print('Validation score for Epoch %d: %f' % (epoch_num +1, val_score))
        with open(logfile, 'a+') as logf:
            logf.write('Validation score for Epoch %d: %f \n' % (epoch_num +int(chkpt)+1, val_score))
        if end_training:
            print('Stopping early at epoch %d: '  % (epoch_num +1))
            with open(logfile, 'a+') as logf:
                logf.write('Stopping early for Epoch %d, with score : %f \n' % (epoch_num +int(chkpt) +1, val_score))
            break
        prev_val_score = val_score
    
    test_score = loaded_model.evaluate(X_test, Y_test, verbose=2)
    print('Test score :', test_score)
    
    with open(logfile, 'a+') as logf:
        logf.write('Test score is : %f \n' % (test_score))

    return

def predict(main_path, test_filelist, model_name, chkpt, timesteps=300,n_input_dim=711, n_output_dim=199):
    data_path= main_path + '/data/'
    save_dir= main_path + '/predicted_feats/' + model_name + '/'
    model_path = main_path + '/models/' + model_name + '/chkpt-'+ str(chkpt)
    #sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=False, clipnorm=0.01)
    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path +'.h5')
    print("Loaded model from disk")
    loaded_model.compile(loss='mean_sum_error', optimizer='adam')
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

if __name__=='__main__':
    option=sys.argv[1]
    main_path='/home/pbaljeka/english-spasm-experiements/slt_experiments/'
    model_name='arctic_100_lstm-func1'
    feature_dimension=[711, 199]
    ext=['.lab', '.cmp']
    batch_size = 128
    nb_epoch = 2
    test_filelist =main_path + '/data/test100_list'
    timesteps=30

    if option == 'train':
        train_func(main_path=main_path, model_name=model_name, batch_size=batch_size, nb_epoch=nb_epoch, timesteps=timesteps)
    elif option == 'retrain':
        retrain(main_path=main_path, model_name=model_name, chkpt=4,batch_size=batch_size, nb_epoch=nb_epoch, timesteps=timesteps)
    elif option == "predict":
        predict(main_path=main_path, test_filelist=test_filelist,model_name=model_name, chkpt=4, timesteps=timesteps)

