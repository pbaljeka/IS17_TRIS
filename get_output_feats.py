import numpy as np
import sys
import os
import binary_io


#This file prepares output features for the neural network utterancewise. Its basically copying the means stored in node_stats as ymean, nmean etc.
#Because some no nodes are missing, i am giving it the yes node means for now.(Basically making an asymmetric node symmetric)- not sure of this right.
io_funcs = binary_io.BinaryIOCollection()
def get_stat(nodename, main_path, feature_dimension, ext):
    io_funcs = binary_io.BinaryIOCollection()
    stat=io_funcs.load_binary_file(main_path +nodename.strip() + ext, feature_dimension)
    return stat

def make_acoustic_tris_feats(tris_file, node, stat):
    io_funcs = binary_io.BinaryIOCollection()
    main_path='/home/pbaljeka/TRIS_Exps3/'
    source_name= 'cmu_us_slt'
    source_path = main_path + source_name + '/festival/coeffs/'
    stats_path = main_path + source_name + '/festival/node_stats/'
    full_path= source_path + tris_file.strip()
    tris_utils= main_path + '/utils/'
    outdir= main_path + source_name + '/festival/nn_tris/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ext=node[0] + stat
    outfile= outdir + tris_file.strip() + '.' + ext
    final_mat=[]
    if node == "yes":
        column_num=1
    elif node== "no":
        column_num=2
    else:
	print("Wrong option")
    collapse_nodes=['pau_3_NNYNNYYYYNYNNYYNNYN','pau_3_NNNYNNYYNYNYYN','pau_3_NNYNNYYYYNYNNYYYYNYN', 's_2_YYYYN']
    with open(full_path + '.tris', 'r') as f:
        for line in f:
            columns=line.strip().split()
            if columns[0][-1] <> "L":
                if columns[int(column_num)] in collapse_nodes:
                    column = columns[int(column_num - 1)]
                else:
                    column = columns[int(column_num)]
                #print(columns[int(column_num)])
                final_mat.append(get_stat(column, stats_path,int(52), '.' + stat))
    #print(final_mat[-1])
    io_funcs.array_to_binary_file(final_mat, outfile)
    return



if __name__=="__main__":
    option=sys.argv[1]
    if option == 'check':
        io_funcs = binary_io.BinaryIOCollection()
        main_path='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/node_stats/'
        make_acoustic_tris_feats('arctic_a0001', 'no', 'std')
    else:
        filelist='/home/pbaljeka/TRIS_Exps3/utils/all_list'
        with open(filelist, 'r') as f:
            for line in f:
                print(line)
                make_acoustic_tris_feats(line.strip(), 'yes', 'mean')
                make_acoustic_tris_feats(line.strip(), 'yes', 'std')
                make_acoustic_tris_feats(line.strip(), 'no', 'mean')
                make_acoustic_tris_feats(line.strip(), 'no', 'std')
