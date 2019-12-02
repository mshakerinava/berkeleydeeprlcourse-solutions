## HW4: Model-Based RL

### Problem 1

**Dynamics Model**

Sometimes the learned dynamics model goes to infinity after 500 timesteps (see for example [this plot](plots/q1/prediction_001.jpg).)  
Predicting `state 17` seems to be hard.

![model predictions](plots/q1/prediction_000.jpg)


### Problem 2

| Policy      | Total Return       |
| ----------- | ------------------ |
| Random      | -151.877 ± 38.3052 |
| Model-Based | 38.8864 ± 20.3343  |


### Problem 3a

**MBRL with on-policy data collection on HalfCheetah"

![HalfCheetah q3a](plots/HalfCheetah_q3_default.jpg)


### Problem 3b

**Hyperparameter sensitivity plots**

![HalfCheetah_q3_actions](plots/HalfCheetah_q3_actions.jpg)
![HalfCheetah_q3_mpc_horizon](plots/HalfCheetah_q3_mpc_horizon.jpg)
![HalfCheetah_q3_nn_layers](plots/HalfCheetah_q3_nn_layers.jpg)
