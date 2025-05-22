#result_conf_matrix.py
import pickle
with open('results/nrec_targets.pickle', 'rb') as handle:
    targets=pickle.load(handle)
    # print(targets)
    handle.close()  
with open('results/nrec_results.pickle', 'rb') as handle:
    rec_results=pickle.load(handle)
    # print(rec_results)
    handle.close()
with open('results/ncat_1.pickle', 'rb') as handle:
    cat_1=pickle.load( handle)
    # print(cat_1)
    handle.close()
    
with open('results/ncat_2.pickle', 'rb') as handle:
    cat_2=pickle.load(handle)
    # print(cat_2)
    handle.close()


with open('results/npath_fns.pickle', 'rb') as handle:
    path_fns=pickle.load(handle)
    # print(path_fns)
    handle.close()

