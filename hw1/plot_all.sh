#!/bin/bash
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python3 plot.py --env $e --hash-bc 90cbb4ee --hash-dagger bd846220 -y "Score Mean" --ylabel Score
done
