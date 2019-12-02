#!/bin/bash
for ((seed=1; seed<=10; seed++))
do
    for env in Humanoid-v2 Hopper-v2 Ant-v2 HalfCheetah-v2 Reacher-v2 Walker2d-v2
    do
        python3 train_bc.py --seed $seed --env $env &
    done
    wait
done
