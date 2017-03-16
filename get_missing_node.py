
def make_list(clbnodefile):
    nodelist=[]
    with open(clbnodefile, 'r') as f:
	for line in f:
	    nodelist.append(line.strip())
    return nodelist

def find_missing(sltnodefile, utils_path, nodelist):
    missing_file=utils_path + '/missing_allnodes'
    with open(sltnodefile, 'r') as f:
	for line in f:
	    if line.strip() not in nodelist:
		with open(missing_file, 'a+') as g:
		    g.write(line)

		
		
if __name__== "__main__":
    main_path='/home/pbaljeka/TRIS_Exps3/'
    utils_path= main_path + '/utils-clb5/'
    slt_nodenames= main_path +'/utils/allnodes'
    clb_nodenames= main_path +'/utils-clb5/allnodes'
    nodelist=make_list(clb_nodenames)
    find_missing(slt_nodenames, utils_path, nodelist)
