import numpy as np
import sys

num_points = sys.argv[1]

vals_pij = []

col_ind_pij = []
rows_pij = []
#edges = []

row_n = 0
with open("rows_"+str(num_points)) as fp1:
    rows = fp1.readlines()

    rows = rows[0].split(" ")
    
    rows = [int(r) for r in rows if r!='']
print(len(rows))
with open("ind_"+str(num_points)) as fp2:
    ind = fp2.readlines()

    ind = ind[0].split(" ")

    ind = [int(i) for i in ind if i!='']

with open("vals_"+str(num_points)) as fp3:
    vals_pij = fp3.readlines()

    #vals_pij = vals[0].split(" ")
    
    #vals_pij = [int(v) for r in vals_pij if r!='']

i1 = 0
edges = {}
with open("pij_all"+str(num_points)+".txt", "w") as fpij, open("edges_"+str(num_points), "w") as eo:
    for p in vals_pij:
        fpij.write(p + " ")
    fpij.write("\n")
    #fpij.write(rows + "\n")
    #fpij.write(ind + "\n")
    offset=0
    while(i1 < len(rows)-1):
        cur_nnz = rows[i1+1] - rows[i1]
        cur_i = 0
        #offset = 0
        while(cur_i < cur_nnz):
            fpij.write(str(i1) + " " + str(ind[i1+offset+cur_i]) + " ")
            
            if(i1 not in edges):
                if(i1 != ind[i1 + offset+ cur_i]):

                    edges[i1] = []
                    if (ind[i1+offset+cur_i] not in edges[i1]):
                        edges[i1].append(ind[i1+offset+cur_i])
                        eo.write(str(i1) + " " + str(ind[i1+offset+cur_i]) + "\n")
                        #fpij.write(str(i1) + " " + str(ind[i1+offset+cur_i]) + " ")

                    if (ind[i1+offset+cur_i] not in edges):
                        edges[ind[i1+offset+cur_i]] = []
                        edges[ind[i1+offset+cur_i]].append(i1)
                        eo.write(str(ind[i1+offset+cur_i]) + " " + str(i1) + "\n")
                    else:
                        edges[ind[i1+offset+cur_i]].append(i1)
                        eo.write(str(ind[i1+offset+cur_i]) + " " + str(i1) + "\n")
            else:
                if(i1 != ind[i1+offset+cur_i]):
                    if (ind[i1+offset+cur_i] not in edges[i1]):
                        edges[i1].append(ind[i1+offset+cur_i])
                        eo.write(str(i1) + " " + str(ind[i1+offset+cur_i]) + "\n")
                    if (ind[i1+offset+cur_i] not in edges):
                        edges[ind[i1+offset+cur_i]] = []
                        edges[ind[i1+offset+cur_i]].append(i1)
                        eo.write(str(ind[i1+offset+cur_i]) + " " + str(i1) + "\n")
                    else:
                        if i1 not in edges[ind[i1+offset+cur_i]]:
                            edges[ind[i1+offset+cur_i]].append(i1)
                            eo.write(str(ind[i1+offset+cur_i]) + " " + str(i1) + "\n")


                  #else:
                     #   edges[ind[i1+offset+cur_i]].append(i1)
                      #  eo.write(str(ind[i1+offset+cur_i]) + " " + str(i1) + "\n")


            #eo.write(str(i1) + " " + str(ind[i1+offset+cur_i]) + "\n") 

            #edges.append(ind[i1 + cur_i])
            
            cur_i += 1
        offset = offset + cur_nnz-1
        i1 += 1
#with open("edges.out2", "w") as eo:
#    for 
#with open("pij_all.txt", "w") as fpij:
 #   fpij.write(vals + "\n")
  #  fpij.write(rows + "\n")
   # fpij.write(ind + "\n")




