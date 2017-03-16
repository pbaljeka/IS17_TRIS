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
def get_missing_nodes(clbnodefile):
    nodelist=[]
    with open(clbnodefile, 'r') as f:
	for line in f:
	    nodelist.append(line.strip())
    return nodelist



def make_acoustic_tris_feats(tris_file, node, stat):
    io_funcs = binary_io.BinaryIOCollection()
    main_path='/home/pbaljeka/TRIS_Exps3/'
    source_name= 'cmu_us_clb'
    source_path = main_path + source_name + '/festival/coeffs5/'
    stats_path = main_path + source_name + '/festival/node_stats/'
    full_path= source_path + tris_file.strip()
    tris_utils= main_path + '/utils-clb5/'
    outdir= main_path + source_name + '/festival/nn_tris5/'
    missing_nodes = get_missing_nodes(tris_utils + 'missing_allnodes')
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
    missing_nodes.extend(collapse_nodes) 
    with open(full_path + '.tris', 'r') as f:
        for line in f:
            columns=line.strip().split()
            if columns[0][-1] <> "L":
                if columns[1] not in missing_nodes and columns[2] not in missing_nodes :
                    column = columns[int(column_num)]
		    #print(columns[0])
                #print(columns[int(column_num)])
                    final_mat.append(get_stat(column, stats_path,int(52), '.' + stat))
    #print(final_mat[-1])
    io_funcs.array_to_binary_file(final_mat, outfile)
    return



if __name__=="__main__":
    filelist='/home/pbaljeka/TRIS_Exps3/utils-clb5/all_list'
    with open(filelist, 'r') as f:
        for line in f:
            print(line)
            make_acoustic_tris_feats(line.strip(), 'yes', 'mean')
            make_acoustic_tris_feats(line.strip(), 'yes', 'std')
            make_acoustic_tris_feats(line.strip(), 'no', 'mean')
            make_acoustic_tris_feats(line.strip(), 'no', 'std')
