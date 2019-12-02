## HW3: Q-Learning and Actor-Critic

### Q-Learning

**Sanity check with Lunar Lander**

![LunarLander plot](plots/lander.svg)
![LunarLander gif](gifs/lunar_lander_dqn.gif)

**Question 1: Basic Q-Learning performance**  
**Question 2: Double Q-Learning**

![Pong plot](plots/pong_dqn.svg)
![Pong gif](gifs/pong-vanilla-dqn.gif)

**Question 3: Experimenting with hyperparameters**

I chose to experiment with the target network update frequency. `--target-update-freq` defines how many main network updates are performed between successive target network updates. Main network updates start after the replay memory has filled up and occur every 4 timesteps.

![update-freq plot](plots/pong_update_freq.svg)


### Actor-Critic

**Question 1: Sanity check with CartPole**

![CartPole plot](plots/cartpole.svg)

**Question 2: Run Actor-Critic with more difficult tasks**

HalfCheetah (hc) and InvertedPendulum (ip).

![HalfCheetah plot](plots/half_cheetah.svg)
![InvertedPendulum plot](plots/inverted_pendulum.svg)
