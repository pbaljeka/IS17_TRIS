import numpy as np
import sys
import os
import binary_io

def one_hot(number, max_size):
    """ Returns the one hot vector of a number, given the max"""
    b = np.zeros(max_size,dtype=float)
    b[number]=1.0
    return b

def get_items(infile):
    """ Returns the statenames/phonenames as a list"""
    with open(infile, 'r') as f:
        itemlist=f.read().strip().split('___')
        itemlist.pop(-1)
        itemlist.insert(0,"0")
    return itemlist
def get_element_one_hot(item, find_list):
    """Returns the item index in a list"""
    if (find_list == 'FLOAT'):
        return [float(item)]
    else:
        return list(one_hot(np.where(np.asarray(find_list)==item)[0][0], len(find_list)))

def get_element(item, find_list):
    """Returns the item index in a list"""
    if (find_list == 'FLOAT'):
        return [float(item)]
    else:
        return np.where(np.asarray(find_list)==item)[0][0]

def get_features(statefile, phonefile):
    """ Returns the list of linguistic features"""
    STATELIST=get_items(statefile)
    PHONELIST=get_items(phonefile)
    POSLIST=[0, 'aux','cc','content','det','in', 'md','pps','to','wp', 'punc']
    POS2LIST=['b','m','e']
    PRESENTLIST=[0, '+','-']
    PLACELIST=[0, 'a','b','d','g','l','p','v', '-']
    PLACE2LIST=[0,'a','d','l','s','-']
    PLACE3LIST=[0,'a','f','l','n','r','s','-']
    POSITIONLIST=[0, 'initial', 'single','final','mid']
    STRENGTHLIST=[0, 1, 2, 3, '-']
    STRENGTH2LIST=[0, '1', '3', '4']
    CODALIST =[0,'coda', 'onset']

    features=[STATELIST,
            STATELIST,
            STATELIST,
            PHONELIST,
            PRESENTLIST,
            PLACE3LIST,
            STRENGTHLIST,
            PLACE2LIST,
            STRENGTHLIST,
            PRESENTLIST,
            PLACELIST,
            PRESENTLIST,
            PHONELIST,
            PRESENTLIST,
            PLACE3LIST,
            STRENGTHLIST,
            PLACE2LIST,
            STRENGTHLIST,
            PRESENTLIST,
            PLACELIST,
            PRESENTLIST,
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            POS2LIST,
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            CODALIST,
            CODALIST,
            CODALIST,
            'FLOAT',
            'FLOAT',
            'FLOAT',
            'FLOAT',
            STRENGTH2LIST,
            STRENGTH2LIST,
            POSITIONLIST,
            PHONELIST,
            PRESENTLIST,
            PLACE3LIST,
            STRENGTHLIST,
            PLACE2LIST,
            STRENGTHLIST,
            PRESENTLIST,
            PLACELIST,
            PRESENTLIST,
            'FLOAT',
            'FLOAT',
            POSLIST,
            POSLIST,
            POSLIST]
    return features
def get_feat_lengths(features):
    feat_lengths=[]
    for feat in features:
        if feat=='FLOAT':
            feat_lengths.append(int(1))
        else:
            feat_lengths.append(len(feat))
    return feat_lengths

def get_feat_position(question_num, statefile, phonefile):
    """ Returns the start_pos, end_pos, and feat_list for given a question_num"""
    features=get_features(statefile, phonefile)
    feat_lengths=get_feat_lengths(features)
    end_positions=np.cumsum(feat_lengths)
    end_pos=end_positions[question_num]
    start_pos=end_pos - feat_lengths[question_num]
    feat_list=features[question_num]
    answer_size=end_positions[-1]
    return start_pos, end_pos, feat_list, answer_size


def get_answer(question, answer, q_list, statefile, phonefile):
    """ Returns the binary answer as  sparse fixed dimensional feature"""
    question_num=get_element(question, q_list)
    start_pos, end_pos, feat_list, answer_size = get_feat_position(question_num, statefile, phonefile)
    #print(start_pos, end_pos, len(feat_list), answer_size)
    full_answer=np.zeros((1, answer_size), dtype='float32')
    a=get_element_one_hot(answer, feat_list)
    full_answer[0,start_pos:end_pos] = np.expand_dims(np.asarray(get_element_one_hot(answer, feat_list)), axis=0)
    return full_answer[0]

def binary_linguistic_feat(linguistic_feat_frame, statefile, phonefile):
    """ Returns the binary linguistc feat"""
    final_list=[]
    features=get_features(statefile, phonefile)
    for feat, list_name in enumerate(features):
        #print "DEBUG: ", feat+2, list_name, feat_list[feat+2]
        final_list.extend(get_element_one_hot(linguistic_feat_frame[feat], list_name))
    #print '\n'.join([ str(element) for element in final_list ])
    return final_list

def question_list(question_list):
    """Returns the question_list"""
    with open(question_list, 'r') as f:
        q_list=[]
        for line in f:
            q_list.append(line.strip())
    return q_list

def binary_operand(operand):
    """Returns the binary operand (3 dim)"""
    return get_element_one_hot(operand, ['<', '>',  'is'])

def make_node_dict(nodelist):
    """This takes the list of nodes and creates a node dict"""
    node_dict={}
    with open(nodelist, 'r') as f:
        for node_num, line in enumerate(f):
            node_dict[line.strip()]=node_num
    return node_dict

def make_node_dict_stat(nodelist, node_stats_path):
    """This takes the list of nodes and creates a node dict, for the mean and std"""
    node_dict_mean={}
    node_dict_std={}
    merlin_io=binary_io.BinaryIOCollection()
    with open(nodelist, 'r') as f:
        for node_num, line in enumerate(f):
            filename=line.strip()
            node_mean=merlin_io.load_binary_file(node_stats_path + filename + '.mean' ,52)
            node_std=merlin_io.load_binary_file(node_stats_path + filename + '.std' ,52)
            node_dict_mean[filename]=node_mean
            node_dict_std[filename]=node_std
    return node_dict_mean, node_dict_std

def get_binary_tris(tris_line, node_dict_mean, node_dict_std, question_list, statenames, phonenames):
    """This takes entire line in a tris file and returns a NN suitable vector"""
    line=tris_line.strip().split()
    node_mean=node_dict_mean[line[0]]
    node_std=node_dict_std[line[0]]
    question=get_element_one_hot(line[3], question_list)
    binary_tris_frame= question
    operand=binary_operand(line[4])
    binary_tris_frame.extend(operand)
    answer=get_answer(line[3], line[5], question_list, statenames, phonenames)
    binary_tris_frame.extend(answer)
    linguistic_feat=binary_linguistic_feat(line[60:], statenames, phonenames)
    binary_tris_frame.extend(linguistic_feat)
    #print(len(binary_tris_frame))
    return  binary_tris_frame, node_mean, node_std

def make_binary_tris_feats(tris_file, indir, outdir):
    io_funcs = binary_io.BinaryIOCollection()
    main_path='/home/pbaljeka/TRIS_Exps3/'
    source_name= 'cmu_us_slt'
    source_path = main_path + source_name + '/festival/'+ indir +'/' #Change the input folder to tris if you want it nodewise else coeffs
    full_path= source_path + tris_file.strip()
    node_stats_path = main_path + source_name + '/festival/node_stats/'
    tris_utils= main_path + '/utils/'
    outdir= main_path + source_name + '/festival/'+ outdir + '/'  #Change the output folder to nn_tris_trees to save the nodewise feats there. else nn_tris
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile= outdir + tris_file.strip()
    nodenames= tris_utils + '/nodenames'
    q_list = question_list(tris_utils + '/question_list')
    statenames = tris_utils + '/statenames'
    phonenames = tris_utils + '/phonenames'
    node_dict_mean, node_dict_std=make_node_dict_stat(nodenames, node_stats_path)
    tris_mat=[]
    mean_mat=[]
    std_mat=[]
    with open(full_path +'.tris', 'r') as f:
        for line in f:
            if line.strip().split()[0][-1] <> "L":
                binary_tris_frame, node_mean, node_std =get_binary_tris(line.strip(), node_dict_mean, node_dict_std, q_list, statenames, phonenames)
                tris_mat.append(binary_tris_frame)
                mean_mat.append(node_mean)
                std_mat.append(node_std)
    io_funcs.array_to_binary_file(tris_mat, outfile + '.tris')
    io_funcs.array_to_binary_file(mean_mat, outfile + '.mean')
    io_funcs.array_to_binary_file(std_mat, outfile + '.std')
    return



if __name__=="__main__":
    option=sys.argv[1]
    utils_path="/home/pbaljeka/TRIS_Exps3/utils/"
    if option=="uttwise":
	indir="coeffs"
	outdir="nn_tris"
	filelist=utils_path + "all_list"
    elif option =="nodewise":
	indir="tris"
	outdir="nn_tirs_trees"
	filelist=utils_path + "all_nodes"
    else:
	print("Wrong option")
	
    with open(filelist, 'r') as f:
	for line in f:
	    print(line)
            make_binary_tris_feats(line.strip(), indir, outdir)
