#result_conf_matrix.py
import pickle, numpy as np
with open('results/nrec_outputs.pickle', 'rb') as handle:
    outputs=pickle.load(handle)
    # print(targets)
    # handle.close()  
with open('results/nrec_results.pickle', 'rb') as handle:
    rec_results=pickle.load(handle)
    # print(rec_results)
    handle.close()
with open('results/ncats.pickle', 'rb') as handle:
    cats_target=pickle.load( handle)
    # print(cats_target)
    handle.close()
    


with open('results/npath_fns.pickle', 'rb') as handle:
    path_fns=pickle.load(handle)
    # print(path_fns)
    handle.close()
    
value_to_find = False
indices = np.where(rec_results == value_to_find)[0]
# print(f"Indices of {value_to_find}: {indices}")

print ("files/path")
for i in indices:
    print(path_fns[i])

print("input2 target")
for i in indices:
    print(cats_target[i])

print("output/results")
print(outputs[indices])
# print(rec_results[indices])

