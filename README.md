# CS394R_Final
Chloe Chen and Logan Persyn

Final Report: `final_report.pdf`

Video: INCLUDE URL

## Environment
1. Create new conda env with python 3.11: `conda create --name env_name python=3.11`
1. Install all dependencies with pip: `pip install -r requirements.txt` or `pip install -r tacc_requirements.txt`
1. Use `pip list --format=freeze > requirements.txt` to create environment.

## Training
- The respective `tacc_train_*` file has the commands and arguments to train agents.

## Testing
- Different noise weight tests can be run with `noise_weight_test_with_*.sh` files. 

## Watching
- An example command to watch the noisy-trained SAC, watch mode is controlled by `--watch` and `--render fps` flags, both must be present:  `python .\tianshou\atari_sac.py --watch --resume-path .\final_policies\sac_trained_on_noisy_env.pth  --logdir ./temp --render 0.01`