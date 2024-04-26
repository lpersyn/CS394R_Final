import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=False, action="store_true")
    parser.add_argument("--noerrorbars", default=False, action="store_true")
    args = parser.parse_args()

    # Load the data
    normal_rainbow_mean_reward = pd.read_csv("./figures/test_results_during_training/mean_reward/normal_rainbow.csv")
    normal_rainbow_mean_reward.rename(columns={"Value": "Normal Rainbow Mean"}, inplace=True)
    normal_rainbow_mean_reward.drop(columns=["Wall time"], inplace=True)
    normal_rainbow_std_reward = pd.read_csv("./figures/test_results_during_training/std_reward/std_rainbow_normal.csv")
    normal_rainbow_std_reward.rename(columns={"Value": "Normal Rainbow Std"}, inplace=True)
    normal_rainbow_std_reward.drop(columns=["Wall time"], inplace=True)
    assert normal_rainbow_mean_reward["Step"].equals(normal_rainbow_std_reward["Step"])

    noisy_rainbow_mean_reward = pd.read_csv("./figures/test_results_during_training/mean_reward/noisy_rainbow.csv")
    noisy_rainbow_mean_reward.rename(columns={"Value": "Noisy Rainbow Mean"}, inplace=True)
    noisy_rainbow_mean_reward.drop(columns=["Wall time"], inplace=True)
    noisy_rainbow_std_reward = pd.read_csv("./figures/test_results_during_training/std_reward/std_rainbow_noisy.csv")
    noisy_rainbow_std_reward.rename(columns={"Value": "Noisy Rainbow Std"}, inplace=True)
    noisy_rainbow_std_reward.drop(columns=["Wall time"], inplace=True)
    assert noisy_rainbow_mean_reward["Step"].equals(noisy_rainbow_std_reward["Step"])


    normal_sac_mean_reward = pd.read_csv("./figures/test_results_during_training/mean_reward/normal_sac.csv")
    normal_sac_mean_reward.rename(columns={"Value": "Normal SAC Mean"}, inplace=True)
    normal_sac_mean_reward.drop(columns=["Wall time"], inplace=True)
    normal_sac_std_reward = pd.read_csv("./figures/test_results_during_training/std_reward/std_sac_normal.csv")
    normal_sac_std_reward.rename(columns={"Value": "Normal SAC Std"}, inplace=True)
    normal_sac_std_reward.drop(columns=["Wall time"], inplace=True)
    assert normal_sac_mean_reward["Step"].equals(normal_sac_std_reward["Step"])

    noisy_sac_mean_reward = pd.read_csv("./figures/test_results_during_training/mean_reward/noisy_sac.csv")
    noisy_sac_mean_reward.rename(columns={"Value": "Noisy SAC Mean"}, inplace=True)
    noisy_sac_mean_reward.drop(columns=["Wall time"], inplace=True)
    noisy_sac_std_reward = pd.read_csv("./figures/test_results_during_training/std_reward/std_sac_noisy.csv")
    noisy_sac_std_reward.rename(columns={"Value": "Noisy SAC Std"}, inplace=True)
    noisy_sac_std_reward.drop(columns=["Wall time"], inplace=True)
    assert noisy_sac_mean_reward["Step"].equals(noisy_sac_std_reward["Step"])

    # max and min steps
    max_steps = max(
        normal_rainbow_mean_reward["Step"].max(),
        noisy_rainbow_mean_reward["Step"].max(),
        normal_sac_mean_reward["Step"].max(),
        noisy_sac_mean_reward["Step"].max(),
    )
    min_steps = min(
        normal_rainbow_mean_reward["Step"].max(),
        noisy_rainbow_mean_reward["Step"].max(),
        normal_sac_mean_reward["Step"].max(),
        noisy_sac_mean_reward["Step"].max(),
    )

    df = pd.DataFrame()
    if args.clip:
        df["Step"] = np.arange(0, min_steps + 1, 100_000)
    else:
        df["Step"] = np.arange(0, max_steps + 1, 100_000)

    df = df.merge(normal_rainbow_mean_reward, on="Step", how="left")
    df = df.merge(normal_rainbow_std_reward, on="Step", how="left")
    df = df.merge(noisy_rainbow_mean_reward, on="Step", how="left")
    df = df.merge(noisy_rainbow_std_reward, on="Step", how="left")
    df = df.merge(normal_sac_mean_reward, on="Step", how="left")
    df = df.merge(normal_sac_std_reward, on="Step", how="left")
    df = df.merge(noisy_sac_mean_reward, on="Step", how="left")
    df = df.merge(noisy_sac_std_reward, on="Step", how="left")

    if args.noerrorbars:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.lineplot(data=df, x="Step", y="Normal Rainbow Mean", ax=ax, label="Normal Rainbow, noise weight = 0.0")
        sns.lineplot(data=df, x="Step", y="Noisy Rainbow Mean", ax=ax, label="Noisy Rainbow, noise weight = 1.0")
        sns.lineplot(data=df, x="Step", y="Normal SAC Mean", ax=ax, label="Normal SAC, noise weight = 0.0")
        sns.lineplot(data=df, x="Step", y="Noisy SAC Mean", ax=ax, label="Noisy SAC, noise weight = 1.0")
        ax.set_xlabel("Training Steps (Millions)")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward During Training")
        ax.legend(loc="upper left")
        plt.show()
    else:

        df["Normal Rainbow Mean-std"] = df["Normal Rainbow Mean"] - df["Normal Rainbow Std"]
        df["Normal Rainbow Mean+std"] = df["Normal Rainbow Mean"] + df["Normal Rainbow Std"]
        df["Noisy Rainbow Mean-std"] = df["Noisy Rainbow Mean"] - df["Noisy Rainbow Std"]
        df["Noisy Rainbow Mean+std"] = df["Noisy Rainbow Mean"] + df["Noisy Rainbow Std"]
        df["Normal SAC Mean-std"] = df["Normal SAC Mean"] - df["Normal SAC Std"]
        df["Normal SAC Mean+std"] = df["Normal SAC Mean"] + df["Normal SAC Std"]
        df["Noisy SAC Mean-std"] = df["Noisy SAC Mean"] - df["Noisy SAC Std"]
        df["Noisy SAC Mean+std"] = df["Noisy SAC Mean"] + df["Noisy SAC Std"]

        # drop the std cols and original mean cols
        df.drop(columns=["Normal Rainbow Std", "Noisy Rainbow Std", "Normal SAC Std", "Noisy SAC Std"], inplace=True)
        df.drop(columns=["Normal Rainbow Mean", "Noisy Rainbow Mean", "Normal SAC Mean", "Noisy SAC Mean"], inplace=True)

        #create df, two cols, steps and mean reward
        normal_rainbow_df = pd.DataFrame()
        normal_rainbow_df["Step"] = pd.concat([df["Step"], df["Step"]])
        normal_rainbow_df["Mean Reward"] = pd.concat([df["Normal Rainbow Mean-std"], df["Normal Rainbow Mean+std"]])
        normal_rainbow_df = normal_rainbow_df.reset_index(drop=True)

        noisy_rainbow_df = pd.DataFrame()
        noisy_rainbow_df["Step"] = pd.concat([df["Step"], df["Step"]])
        noisy_rainbow_df["Mean Reward"] = pd.concat([df["Noisy Rainbow Mean-std"], df["Noisy Rainbow Mean+std"]])
        noisy_rainbow_df = noisy_rainbow_df.reset_index(drop=True)

        normal_sac_df = pd.DataFrame()
        normal_sac_df["Step"] = pd.concat([df["Step"], df["Step"]])
        normal_sac_df["Mean Reward"] = pd.concat([df["Normal SAC Mean-std"], df["Normal SAC Mean+std"]])
        normal_sac_df = normal_sac_df.reset_index(drop=True)

        noisy_sac_df = pd.DataFrame()
        noisy_sac_df["Step"] = pd.concat([df["Step"], df["Step"]])
        noisy_sac_df["Mean Reward"] = pd.concat([df["Noisy SAC Mean-std"], df["Noisy SAC Mean+std"]])
        noisy_sac_df = noisy_sac_df.reset_index(drop=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.lineplot(data=normal_rainbow_df, x="Step", y="Mean Reward", ax=ax, label="Normal Rainbow, noise weight = 0.0", errorbar=("pi", 100))
        sns.lineplot(data=noisy_rainbow_df, x="Step", y="Mean Reward", ax=ax, label="Noisy Rainbow, noise weight = 1.0", errorbar=("pi", 100))
        sns.lineplot(data=normal_sac_df, x="Step", y="Mean Reward", ax=ax, label="Normal SAC, noise weight = 0.0", errorbar=("pi", 100))
        sns.lineplot(data=noisy_sac_df, x="Step", y="Mean Reward", ax=ax, label="Noisy SAC, noise weight = 1.0", errorbar=("pi", 100))
        ax.set_xlabel("Training Steps (Millions)")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward During Training")
        ax.legend(loc="upper left")
        plt.show()

