import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def array(list):
    return np.array(list)

def get_all_dicts(input_path):
    noise_levels = {}
    for dir in os.listdir(input_path):
        file_path = os.path.join(input_path, dir)
        if not os.path.isdir(file_path):
            continue
        #get the noise level
        noise = float(dir.split('_')[1] + '.' + dir.split('_')[2])
        # traverse all dirs until no more dirs
        subdir = os.listdir(file_path)[0]
        while True:
            file_path = os.path.join(file_path, subdir)
            # if there are no more dirs in subdir, break
            subsubdirs = [name for name in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, name))]
            total_dirs = len(subsubdirs)
            if total_dirs == 0:
                break
            subdir = subsubdirs[0]
        watch_result_path = os.path.join(file_path, 'watch_result.txt')
        #read a dict from the file
        with open(watch_result_path, 'r') as f:
            watch_result = eval(f.read())
            noise_levels[noise] = watch_result
    return noise_levels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=str, required=True)
    parser.add_argument("--input1-name", type=str, required=True)
    parser.add_argument("--input2", type=str, required=True)
    parser.add_argument("--input2-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    input1 = get_all_dicts(args.input1)
    input2 = get_all_dicts(args.input2)

    assert input1.keys() == input2.keys()

    temp = []
    for noise in input1.keys():
        for ret1,ret2 in zip(input1[noise]['returns'], input2[noise]['returns']):
            temp.append([noise, ret1, ret2])

    df = pd.DataFrame(temp, columns=['noise', args.input1_name, args.input2_name])
    
    dd=pd.melt(df,id_vars=['noise'],value_vars=[args.input1_name, args.input2_name], var_name='models')
    sns.boxplot(x='noise',y='value',data=dd,hue='models')
    # Get the x-axis labels
    x_labels = plt.xticks()[0]

    # Filter the x-axis labels to show every other label
    x_labels = x_labels[::2]
    plt.xticks(x_labels)
    plt.ylabel('Returns')
    plt.xlabel('Noise Weight')
    plt.title(f"{args.model_name} evaluated on different noise weights")
    # Set the x-axis labels
    plt.show()

    # # plot the results
    # labels = [noise for noise in sorted(noise_levels.keys())]
    # returns = [noise_levels[noise]['returns'] for noise in labels]
    # # print(returns)
    # fig, ax = plt.subplots()
    # ax.boxplot(returns, labels=labels)
    # ax.set_title('Returns for different noise levels')
    # ax.set_xlabel('Noise level')
    # ax.set_ylabel('Returns')
    # plt.show()