import os
import sys
import numpy as np
import binary_io

io_funcs=binary_io.BinaryIOCollection()
#feature_file for non-terminal  is:
#root_node, left_node, right_node,-3
#question, operator, operand-3
#root node name -1
#f0, acoustic feature, voicing - 1+ 50 + 1
#frame number, linguistic feature- 1+65
#TOTAL=123
def interleave(mean_vec,std_vec, outfile):
    interleave_vec =np.vstack((mean_vec,std_vec)).reshape((-1,),order='F')
    #print(interleave_vec.shape)
    with open(outfile, 'a+') as f:
        np.savetxt(f, interleave_vec.reshape(1,interleave_vec.shape[0]), fmt='%.7f')
        #np.savetxt(f,  interleave_vec[None,:], delimiter='ii', newline='\n', fmt='%10.5f')

def load_data(data_path, filename,):
    merlin_io= binary_io.BinaryIOCollection()
    mean_mat=merlin_io.load_binary_file(data_path + filename + '.mean',int(52))
    assert mean_mat.shape[1]==52, "Data loading improper"
    std_mat=merlin_io.load_binary_file(data_path + filename + '.std',int(52))
    assert std_mat.shape[1]==52, "Data loading improper"
    return mean_mat, std_mat
def load_rawparams_std(rawparamsfile):
    startframe=1
    with open(rawparamsfile, 'r') as f:
        for line in f:
            line_frame=line.strip().split()[1::2]
            std_frame=[]
            for i in line_frame:
                std_frame.append(float(i))
            if startframe==1:
                data_mat=np.asarray(np.expand_dims(std_frame, axis=0), dtype='float32')
                startframe=0
            else:
                data_mat=np.r_[data_mat, np.asarray(np.expand_dims(std_frame, axis=0), dtype='float32')]
    assert data_mat.shape[1]==52, "Data loading improper"
    return data_mat
def load_rawparams_mean(rawparamsfile):
    startframe=1
    with open(rawparamsfile, 'r') as f:
        for line in f:
            line_frame=line.strip().split()[::2]
            mean_frame=[]
            for i in line_frame:
                mean_frame.append(float(i))
            if startframe==1:
                data_mat=np.asarray(np.expand_dims(mean_frame, axis=0), dtype='float32')
                startframe=0
            else:
                data_mat=np.r_[data_mat, np.asarray(np.expand_dims(mean_frame, axis=0), dtype='float32')]
    assert data_mat.shape[1]==52, "Data loading improper"
    return data_mat


def make_senone_dict(senone_file):
    s_dict={}
    with open(senone_file, 'r') as f:
        for line in f:
            s_line=line.strip().split()
            s_dict[int(s_line[0])]=s_line[1]
    return s_dict

def calc_stats(filename, indir, outfile, std_vec):
    mean_mat, std_mat=load_data(indir, filename)
    mean_vec=np.mean(mean_mat,axis=0)

    #io_funcs.array_to_binary_file(mean_vec, outfile +'.mean')
    assert mean_vec.shape[0] == 52, "Mean calculation is wrong"
    std_vec=np.std(std_mat, axis=0)
    assert std_vec.shape[0] == 52, "Std calculation is wrong"
    #io_funcs.array_to_binary_file(std_vec, outfile +'.std')
    interleave(mean_vec,std_vec, outfile)

def make_rawparams_file(indir, outfile, s_dict, num_senones, true_mean, true_std ):
    """This is to handle the line numbers properly"""
    for i in range(num_senones):
        if i in s_dict.keys():
            calc_stats(s_dict[i], indir, outfile, true_std[i])
        else:
            print(i)
            interleave(true_mean[i], true_std[i], outfile)


if __name__=="__main__":
    main_path='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/'
    indir=main_path +'/predicted_norm_nn_tris_trees/un_norm_slt_tris_norm_sgd-mean_sum_error/'
    outfile='cmu_us_slt_mcep.rawparams'
    filelist='/home/pbaljeka/TRIS_Exps3/utils/senones'
    rawparams='/home/pbaljeka/TRIS_Exps3/utils/cmu_us_slt_mcep.rawparams'
    s_number='/home/pbaljeka/TRIS_Exps3/utils/senone_numbers'
    true_std=load_rawparams_std(rawparams)
    true_means=load_rawparams_mean(rawparams)
    #print(std_vec[7])
    s_dict=make_senone_dict(s_number)
    print s_dict[41]
    num_senones=max(s_dict.keys())
    print(num_senones)
    make_rawparams_file(indir, outfile, s_dict, num_senones, true_means, true_std)

