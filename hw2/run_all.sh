#!/bin/bash

mkdir -p plots


#################
### Problem 4 ###
#################

python3 train_pg_f18.py -l 1 -s 32 CartPole-v0 -n 100 -b 1000 -e 10 -dna --exp_name sb_no_rtg_dna
python3 train_pg_f18.py -l 1 -s 32 CartPole-v0 -n 100 -b 1000 -e 10 -rtg -dna --exp_name sb_rtg_dna
python3 train_pg_f18.py -l 1 -s 32 CartPole-v0 -n 100 -b 1000 -e 10 -rtg --exp_name sb_rtg_na
python3 train_pg_f18.py -l 1 -s 32 CartPole-v0 -n 100 -b 5000 -e 10 -dna --exp_name lb_no_rtg_dna
python3 train_pg_f18.py -l 1 -s 32 CartPole-v0 -n 100 -b 5000 -e 10 -rtg -dna --exp_name lb_rtg_dna
python3 train_pg_f18.py -l 1 -s 32 CartPole-v0 -n 100 -b 5000 -e 10 -rtg --exp_name lb_rtg_na

python3 plot.py data/sb_*
mv plot.svg plots/p4_sb.svg

python3 plot.py data/lb_*
mv plot.svg plots/p4_lb.svg


#################
### Problem 5 ###
#################

python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 100 -lr 0.1 -rtg --exp_name ip_b100_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 300 -lr 0.1 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 1000 -lr 0.1 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 3000 -lr 0.1 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 10000 -lr 0.1 -rtg --exp_name ip_b10000_lr0.1

python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 100 -lr 0.03 -rtg --exp_name ip_b100_lr0.01
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 300 -lr 0.03 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 1000 -lr 0.03 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 3000 -lr 0.03 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 10000 -lr 0.03 -rtg --exp_name ip_b10000_lr0.01

python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 100 -lr 0.01 -rtg --exp_name ip_b100_lr0.01
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 300 -lr 0.01 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 1000 -lr 0.01 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 3000 -lr 0.01 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 10000 -lr 0.01 -rtg --exp_name ip_b10000_lr0.01

python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 100 -lr 0.003 -rtg --exp_name ip_b100_lr0.01
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 300 -lr 0.003 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 1000 -lr 0.003 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 3000 -lr 0.003 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 10000 -lr 0.003 -rtg --exp_name ip_b10000_lr0.01

python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 100 -lr 0.001 -rtg --exp_name ip_b100_lr0.001
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 300 -lr 0.001 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 1000 -lr 0.001 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 3000 -lr 0.001 -rtg --exp_name ip_b1000_lr0.1
python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 10000 -lr 0.001 -rtg --exp_name ip_b10000_lr0.001

python3 plot.py data/ip_*
mv plot.svg plots/p5.svg


#################
### Problem 7 ###
#################

python3 train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005

python3 plot.py data/ll_b40000_r0.005*
mv plot.svg plots/p7.svg


#################
### Problem 8 ###
#################

python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b10000_r0.005
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.01  -rtg --nn_baseline --exp_name hc_b10000_r0.01
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.02  -rtg --nn_baseline --exp_name hc_b10000_r0.02
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b30000_r0.005
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.01  -rtg --nn_baseline --exp_name hc_b30000_r0.01
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02  -rtg --nn_baseline --exp_name hc_b30000_r0.02
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b50000_r0.005
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.01  -rtg --nn_baseline --exp_name hc_b50000_r0.01
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02  -rtg --nn_baseline --exp_name hc_b50000_r0.02

python3 plot.py data/hc_b*
mv plot.svg plots/p8a.svg

python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --exp_name hc_dc0.95_b50000_r0.02
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --exp_name hc_dc0.95_rtg_b50000_r0.02
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline --exp_name hc_dc0.95_bl_b50000_r0.02
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_dc0.95_rtg_bl_b50000_r0.02

python3 plot.py data/hc_d*
mv plot.svg plots/p8b.svg
