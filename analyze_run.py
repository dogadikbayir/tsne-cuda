import sys
import os
import matplotlib.pyplot as plt

def read_file(filename):
  lines = []
  with open(filename, 'r') as f1:
  
    lines = f1.readlines()
 
  grads = []
  time_attrs = []
  time_reps = []
  time_totals = []
  time_reorder = 0
  time_perm = 0
  for line in lines:
    if len(line.split('_')) >= 3:
      #print(line.split('_'))
      if(line.split('_')[2]).split()[0] == 'perm:':
        time_perm = float((line.split(':')[1])[0:len(line.split(':')[1])-2])
      if(line.split('_')[2]).split()[0] == 'reorder:':
        time_reorder = float((line.split(':')[1])[0:len(line.split(':')[1])-2])
      if (line.split('_')[2]).split()[0] == 'attr:':
        print(time_perm)
        print(time_reorder)
        time_attrs.append(float((line.split(':')[1])[0:len(line.split(':')[1])-2]) + time_perm + time_reorder)
      elif (line.split('_')[2]).split()[0]  == 'nbodyfft:':
        time_reps.append(float((line.split(':')[1])[0:len(line.split(':')[1])-2]))
    if len(line.split('_')) == 2:
      if (line.split('_')[0])  == 'total':
        time_totals.append(float((line.split(':')[1])[0:len(line.split(':')[1])-2]))
  
    if len(line.split(' ')) >= 3:
      #print(line.split(' '))
      if line.split(' ')[2] == 'Avg.':
        grads.append(float(line.split(':')[1]))

  #plt.figure(figsize=(16,6))

  #plt.plot(range(len(time_attrs)), time_attrs)
  #plt.savefig('out.png')
  new_time_attrs = []
  new_time_reps = []
  new_time_tots = []
  i = 0
  for a in time_attrs:
    if i % 2 == 0:
      new_time_attrs.append(a)
    i += 1
  i=0
  for t in time_totals:
    if i % 2 == 0:
      new_time_tots.append(t)
    i+=1
  i=0
  for r in time_reps:
    if i%2==0:
      new_time_reps.append(r)
  return grads, new_time_attrs, new_time_reps, new_time_tots

#print(time_attrs)
#print(time_totals)
#print(time_reps)
#print(time_totals)


#main
#first read the vanilla file
#v_grads, v_attrs, v_reps, v_tots = read_file('vanilla/prb20_v100_D8M_10k_rab.out')
r_grads, r_attrs, r_reps, r_tots = read_file('mnist.run')

#plot the perf 
plt.figure(figsize=(16,6))
#plt.plot(range(len(v_attrs)), v_attrs, label='Vanilla Attractive Forces')
plt.plot(range(len(r_attrs)), r_attrs, label='Rabbit Attractive Forces')
plt.legend()
plt.savefig('attr.png')

plt.figure(figsize=(16,6))
#plt.plot(range(len(v_reps)), v_reps, label='Vanilla Repulsive Forces')
plt.plot(range(len(r_reps)), r_reps, label='Rabbit Repulsive Forces')
plt.legend()
plt.savefig('rep.png')

plt.figure(figsize=(16,6))
#plt.plot(range(len(v_tots)), v_tots, label='Vanilla Total Time')
plt.plot(range(len(r_tots)), r_tots, label='Rabbit Total Time')
plt.legend()
plt.savefig('total.png')

plt.figure(figsize=(16,6))
plt.plot([i*10 for i in range(len(r_grads))], r_grads, label='MNist Gradients')
plt.legend()
plt.savefig('grads.png')
