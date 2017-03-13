 ###normalisation information
import logging
import os
from itertools import islice, chain
import numpy as np
from merlin_gen_scripts.acoustic_composition import AcousticComposition
from merlin_gen_scripts.min_max_norm import MinMaxNormalisation
import sys
from merlin_gen_scripts.mean_variance_norm import MeanVarianceNorm
import binary_io
def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    with open(file_id_list, 'r') as fileid_list:
        for file_id in fileid_list:
            file_name = file_dir + '/' + file_id.strip() + file_extension
            file_name_list.append(file_name)
    return  file_name_list

def batchgen(iterable, size):#This  takes the list of chuked filenames and returns the
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        #yield batchiter
        yield chain([batchiter.next()], batchiter)


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
    return data_mat, data_mat.shape[0]

def calc_mean(data_path, filelist, ext, chunksize=100, feature_dimension=52):
    file_seq = open(filelist, 'r')
    temp_sum=np.zeros((int(feature_dimension)))
    num_frames=0
    for filenames_chunk in batchgen(file_seq, chunksize):
        data_mat, frames=load_batch_datasets(data_path, list(filenames_chunk), int(feature_dimension), ext)
        temp_sum=temp_sum + np.sum(data_mat, axis=0)
        num_frames += frames
    print(num_frames)
    return temp_sum / num_frames

def calc_std(data_path, filelist, ext, mean, chunksize=100, feature_dimension=52):
    file_seq = open(filelist, 'r')
    temp_std=np.zeros((int(feature_dimension)))
    num_frames=0
    for filenames_chunk in batchgen(file_seq, chunksize):
        data_mat, frames=load_batch_datasets(data_path, list(filenames_chunk), int(feature_dimension), ext)
        temp_std=temp_std + np.sum(((data_mat - np.tile(mean,(frames,1)))**2), axis=0)
        num_frames += frames
    print num_frames
    return np.sqrt(temp_std / num_frames)

def calc_acoustic_stats(data_path, utils_dir,  chunksize=100, feature_simension=52):
    filelist= utils_dir + '/all_list'
    merlin_io=binary_io.BinaryIOCollection()
    for ext in ['mean', 'std']:
        print('Calculating stats for : ', ext)
        mean_vec=calc_mean(data_path, filelist,  '.'+ext)
        std_vec = calc_std(data_path, filelist, '.'+ext, mean_vec)
        merlin_io.array_to_binary_file(mean_vec, utils_dir +  ext + '.mean')
        merlin_io.array_to_binary_file(std_vec, utils_dir + ext + '.std')

def do_lab_normalization(data_dir, utils_dir, n_input_dim=1488):
    logger=logging.getLogger("label_normalization")
    logger.info('Doing input feature normaization')
    all_list=utils_dir +'/all_list'
    label_norm_file = utils_dir + '/label_norm.dat'
    binary_label_dir= data_dir + '/nn_tris'
    normalized_label_dir=data_dir + '/norm_nn_tris'
    if not os.path.exists(normalized_label_dir):
        os.makedirs(normalized_label_dir)

    binary_label_all_list   = prepare_file_path_list(all_list, binary_label_dir, '.tris')
    label_norm_all_list  = prepare_file_path_list(all_list, normalized_label_dir, '.tris')
    min_max_normaliser = MinMaxNormalisation(feature_dimension = n_input_dim, min_value = 0.01, max_value = 0.99)
    min_max_normaliser.find_min_max_values(binary_label_all_list)
    min_max_normaliser.normalise_data(binary_label_all_list, label_norm_all_list)
    label_min_vector = min_max_normaliser.min_vector
    label_max_vector = min_max_normaliser.max_vector
    label_norm_info = np.concatenate((label_min_vector, label_max_vector), axis=0)
    label_norm_info = np.array(label_norm_info, 'float32')
    fid = open(label_norm_file, 'wb')
    label_norm_info.tofile(fid)
    fid.close()
    logger.info('saved %s vectors to %s' %(label_min_vector.size, label_norm_file))

def do_lab_normalization_prediction(data_dir, utils_dir, n_input_dim=1488):
    logger=logging.getLogger("label_normalization")
    logger.info('Doing input feature normaization')
    all_list=utils_dir +'/nodenames'
    label_norm_file = utils_dir + '/label_norm.dat'
    binary_label_dir= data_dir + '/nn_tris_trees'
    normalized_label_dir=data_dir + '/norm_nn_tris_trees'
    if not os.path.exists(normalized_label_dir):
        os.makedirs(normalized_label_dir)
    binary_label_all_list   = prepare_file_path_list(all_list, binary_label_dir, '.tris')
    label_norm_all_list  = prepare_file_path_list(all_list, normalized_label_dir, '.tris')
    min_max_normaliser = MinMaxNormalisation(feature_dimension = n_input_dim, min_value = 0.01, max_value = 0.99)
    min_max_normaliser.load_min_max_values(label_norm_file)
    min_max_normaliser.normalise_data(binary_label_all_list, label_norm_all_list)

def normalize_data(data_path, filename, mean, std, ext, feature_dimension=52):
    merlin_io=binary_io.BinaryIOCollection()
    data_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
    norm_data =  (data_mat - np.tile(mean, (data_mat.shape[0],1)))/ (np.tile(std, (data_mat.shape[0],1)) + 0.00000001)
    return norm_data

def denormalize_data(data_path, filename, mean, std, ext, feature_dimension=52):
    merlin_io=binary_io.BinaryIOCollection()
    data_mat=merlin_io.load_binary_file(data_path + filename.strip() + ext, feature_dimension)
    un_norm_data =  (data_mat* (np.tile(std, (data_mat.shape[0],1))+ 0.00000001)) + np.tile(mean, (data_mat.shape[0],1))
    return un_norm_data

def do_acoustic_normalisation(data_dir, utils_dir, n_input_dim=52):
    logger=logging.getLogger("acoustic_normalization")
    logger.info('normalising acoustic (output) features using MVN')
    all_list=utils_dir +'/all_list'
    in_data_dir= data_dir + '/nn_tris/'
    out_data_dir=data_dir + '/norm_nn_tris/'
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    merlin_io=binary_io.BinaryIOCollection()
    for stat in ['mean', 'std']:
        with open(all_list, 'r') as f:
            mean=merlin_io.load_binary_file(utils_dir + stat  + '.mean', int(52))
            std=merlin_io.load_binary_file(utils_dir + stat  + '.std', int(52))
            for filename in f:
		for ext in ['.mean', '.std', '.ymean', '.ystd', '.nmean', '.nstd']:
                    norm_data=normalize_data(in_data_dir, filename.strip(), mean, std, ext, feature_dimension=52)
                    outfile= out_data_dir + filename.strip() + ext
                    merlin_io.array_to_binary_file(norm_data, outfile)

def do_acoustic_normalisation_prediction(data_dir, utils_dir, n_input_dim=52):
    logger=logging.getLogger("acoustic_normalization")
    logger.info('normalising acoustic (output) features using MVN')
    all_list=utils_dir +'/nodenames'
    in_data_dir= data_dir + '/nn_tris_trees/'
    out_data_dir=data_dir + '/norm_nn_tris_trees/'
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    merlin_io=binary_io.BinaryIOCollection()
    for stat in ['mean', 'std']:
        with open(all_list, 'r') as f:
            mean=merlin_io.load_binary_file(utils_dir + stat  + '.mean', int(52))
            std=merlin_io.load_binary_file(utils_dir + stat  + '.std', int(52))

            for filename in f:
                for ext in ['mean', 'std']:
                    norm_data=normalize_data(in_data_dir, filename.strip(), mean, std, '.'+ext, feature_dimension=52)
                    outfile= out_data_dir + filename.strip() + '.' + ext
                    merlin_io.array_to_binary_file(norm_data, outfile)


def do_acoustic_denormalisation_prediction(data_dir, utils_dir, model_name, n_input_dim=52):
    logger=logging.getLogger("acoustic_denormalization")
    logger.info('normalising acoustic (output) features using MVN')
    all_list=utils_dir +'/senones'
    in_data_dir= data_dir + '/predicted_norm_nn_tris_trees/' + model_name + '/'
    out_data_dir=data_dir + '/predicted_norm_nn_tris_trees/un_norm_' + model_name + '/'
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    merlin_io=binary_io.BinaryIOCollection()
    for stat in ['mean', 'std']:
        with open(all_list, 'r') as f:
            mean=merlin_io.load_binary_file(utils_dir + stat  + '.mean', int(52))
            std=merlin_io.load_binary_file(utils_dir + stat  + '.std', int(52))
            for filename in f:
                denorm_data=denormalize_data(in_data_dir, filename.strip(), mean, std, '.'+stat, feature_dimension=52)
                outfile= out_data_dir + filename.strip() + '.' + stat
                merlin_io.array_to_binary_file(denorm_data, outfile)

if __name__=="__main__":
    option=sys.argv[1]
    data_dir="/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/"
    utils_dir="/home/pbaljeka/TRIS_Exps3/utils/"
    model_name='slt_tris_norm_AD_1'
    if option=="normalize_train":
	do_lab_normalization(data_dir, utils_dir, int(1488))
    	calc_acoustic_stats(data_dir + '/nn_tris/', utils_dir)
	do_acoustic_normalisation(data_dir, utils_dir)
    elif option=="normalize_predict":
	do_lab_normalization_prediction(data_dir, utils_dir, int(1488))
	do_acoustic_normalisation_prediction(data_dir, utils_dir)
    elif option =="denormalize":
	do_acoustic_denormalisation_prediction(data_dir, utils_dir, model_name)

