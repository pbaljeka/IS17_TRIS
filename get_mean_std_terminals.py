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
def load_data(filename, indir):
    with open(indir +'/'+ filename + '.tris', 'r') as f:
        startfile=1
        for line in f:
            line_frame=line.strip().split()[4:56]
            mcep_frame=[]
            for i in line_frame:
                mcep_frame.append(float(i))
            mcep_frame[-1]=mcep_frame[-1]/10.0
            if startfile==1:
                data_mat=np.asarray(np.expand_dims(mcep_frame, axis=0), dtype='float32')
                startfile=0
            else:
                data_mat=np.r_[data_mat, np.asarray(np.expand_dims(mcep_frame, axis=0), dtype='float32')]
    assert data_mat.shape[1]==52, "Data loading improper"
    return data_mat

def calc_stats(filename, indir, outdir):
    data_mat=load_data(filename, indir)
    outfile= outdir +'/'+filename
    mean_vec=np.mean(data_mat,axis=0)
    io_funcs.array_to_binary_file(mean_vec, outfile +'.mean')
    assert mean_vec.shape[0] == 52, "Mean calculation is wrong"
    std_vec=np.std(data_mat, axis=0)
    assert std_vec.shape[0] == 52, "Std calculation is wrong"
    io_funcs.array_to_binary_file(std_vec, outfile +'.std')

if __name__=="__main__":
    main_path='/home/pbaljeka/TRIS_Exps2/cmu_us_slt/festival/'
    indir=main_path +'/tris_terminals/'
    outdir=main_path + '/binary_tris_terminal_stats/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    filelist='/home/pbaljeka/TRIS_Exps2/cmu_us_slt-tris_utils/terminals_filelist'
    with open(filelist, 'r') as f:
        for filename in f:
            print(filename)
            calc_stats(filename.strip(), indir, outdir)

