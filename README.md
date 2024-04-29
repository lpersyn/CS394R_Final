# CS394R_Final
Chloe Chen and Logan Persyn

Final Report: `final_report.pdf`

Video: [YouTube](https://youtu.be/ae6hiMKkm5w)


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

## RAM
- All code and results from RAM model training are in the ram folder. To train the model, use `python sac_ram.py --train`. 500 episodes should take around 2-4 hours to run depending on your computation power.
- `python sac_ram.py --test` can be used to test already trained model.
