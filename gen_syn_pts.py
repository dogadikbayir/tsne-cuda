import numpy as np
import sys
import os

#import faiss
import time
from tsnecuda import TSNE

from sklearn.datasets import make_classification

num_points = int(sys.argv[1])
num_dims = int(sys.argv[2])
sep = int(sys.argv[3])
k = int(sys.argv[4])

option = int(sys.argv[5])
reorder = int(sys.argv[6])

x, y = make_classification(n_samples=num_points, n_features=num_dims,
    n_redundant=int(num_dims/2), n_informative=int(num_dims/2), class_sep=sep,
    n_clusters_per_class=1, scale=10.0,
    n_classes=4,shuffle=True,random_state=42)

x = x.astype('float32')

if option == 0:
  np.savetxt(str(num_points) + ".data", x, delimiter=" ", fmt='%f')

elif option == 1:
  X_emb = TSNE(verbose=True).fit_transform(x)

else:
  print("Inside faiss branch")
  start = time.perf_counter()
  res = faiss.StandardGpuResources()

  index_flat = faiss.index_factory(num_dims, "IVF4096,PQ64")
  #gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
  co = faiss.GpuClonerOptions()
  
  gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat, co)

  #index.train(x)
  print("train")
  start_train = time.perf_counter()
  gpu_index_flat.train(x)
  print("Train time: " + str(time.perf_counter()-start_train) + "s")
  
  print("add index")
  start_add = time.perf_counter()
  gpu_index_flat.add(x)
  print("Add time: " + str(time.perf_counter()-start_add) + "s")

  #gpu_index_flat.search(x[:5], 123)

  
  #nlist=20
  
    #quantizer = faiss.IndexFlatL2(num_dims)
  #index = faiss.IndexIVFFlat(quantizer, num_dims, nlist)
  
  #index.nprobe = 20

  #index.train(x)
  #index.add(x)

  #print(index.is_trained)
  #print(index.ntotal)
  print("Starting search:")
  start_search = time.perf_counter()
  D, I = gpu_index_flat.search(x, k)
  print("Search time: " + str(time.perf_counter()-start_search) + "s")

  print (type(I))
  #print (D)

  print("Total Elapsed time: " + str(time.perf_counter()-start) + "s")

  #print (I)
  np.savetxt("faiss_res.txt", I, delimiter=" ")


