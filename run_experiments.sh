#!/bin/bash
num_points=500000
for i in {1..7}
do
  echo "Num points: $num_points"

  python gen_syn_pts.py $num_points 50 15 150 1 1

  num_points=($num_points*2)
done
