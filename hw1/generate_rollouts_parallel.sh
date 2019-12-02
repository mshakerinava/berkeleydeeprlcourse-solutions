#!/bin/bash
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python3 run_expert.py experts/$e.pkl $e --save --seed 123 --num_rollouts 20 &
done
wait
