# JSBSim Gym Environment

![image](sample_animation.gif)

## Usage

You can train a Soft Actor-Critic (SAC) or Proximal Policy Optimization (PPO) agent on the environment by simply running:
```
python train.py
```
Once the agent is trained, you can watch it fly to a random goal using:
```
python test.py
```


## Important Files

The main files defining the environment and feature transformation are `jsbsim_gym/jsbsim_gym.py` and `jsbsim_gym/features.py`. The files under `jsbsim_gym/visualization` are auxiliary files for rendering the environment. 

- `jsbsim_gym.py`: This file defines the environment which wraps a JSBSim simulation which runs an F-16 aerodynamics model. The environment class defines a goal and reward function for the agent. Additional shaping rewards are also defined in a Gym wrapper in this file. 
- `features.py`: This file defines a feature extractor for the JSBSim environment. This is the feature vector I found to be most beneficial for this task. Further details can be found in the comments in this file.
- `train.py`: This is a short script for training a SAC agent on the JSBSim environment. The hardcoded parameters should be sufficient to get decent results. The script takes about 12 hours to run on my desktop though time my vary depending on hardware.
- `test.py`: This script will run the trained agent for one episode while visualizing the environment. The visualization will automatically be saved to an MP4 video and GIF animation.
