#!/bin/bash

mkdir -p output

#=== Part1 ===#

#= Q1 =#
python3 run_dqn_atari.py > output/p1_q1.txt

#= Q2 =#
python3 run_dqn_atari.py --double > output/p1_q2.txt

#= Q3 =#
python3 run_dqn_atari.py --target-update-freq 1e3 > output/p1_q3a.txt
python3 run_dqn_atari.py --target-update-freq 3e3 > output/p1_q3b.txt
python3 run_dqn_atari.py --target-update-freq 3e4 > output/p1_q3c.txt


#=== Part2 ===#

#= Q1 =#
python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1
python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu 100 -ngsptu 1
python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1 -ngsptu 100
python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu 10 -ngsptu 10

#= Q2 =#
python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name ip_10_10 -ntu 10 -ngsptu 10
python3 train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name hc_10_10 -ntu 10 -ngsptu 10
