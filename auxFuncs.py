import matplotlib.pyplot as plt
import os

def env_max_score(env_name):
    if env_name == "CartPole-v0":
        return 200
    elif env_name == "CartPole-v1":
        return 500
    elif env_name == "MountainCar-v0":
        return -110
    elif env_name == "LunarLander-v2":
        return 200
    elif env_name == "Acrobot-v1":
        return -80

def env_sol_score(env_name):
    if env_name == "CartPole-v0":
        return 195
    elif env_name == "CartPole-v1":
        return 487.5
    elif env_name == "MountainCar-v0":
        return -112.75
    elif env_name == "LunarLander-v2":
        return 200
    elif env_name == "Acrobot-v1":
        return -80

def appender(input, output):
    for i in range(0, len(output)):
        output[i].append(input[i])
    return output

def multicolor_plotter(episodes, scores, desired_score, train_done, mean_score, learning_rate, desired_env, inverse_train_done):
    fig = plt.figure()  # creates the figure
    ax3 = fig.add_subplot(1, 1, 1)

    scores_train = zero_to_nan([a*b for a,b in zip(scores,train_done)])
    episodes_train = zero_to_nan([a*b for a,b in zip(episodes,train_done)])
    mean_score_train = zero_to_nan([a*b for a,b in zip(mean_score,train_done)])

    scores_pre_train = zero_to_nan([a*b for a,b in zip(scores,inverse_train_done)])
    episodes_pre_train = zero_to_nan([a*b for a,b in zip(episodes,inverse_train_done)])
    mean_score_pre_train = zero_to_nan([a*b for a,b in zip(mean_score,inverse_train_done)])


    # These plots are after training has started
    ax3.scatter(episodes_train, scores_train, facecolor='blue', alpha=0.15,
                label = 'Score while training')  # plots the agent score
    ax3.plot(episodes_train, mean_score_train, linestyle = '-', color = 'red', lw=0.4,
             label = 'Mean score while training')  # plots the mean score of the last 100 eposides

    # These plots are before training has started
    ax3.scatter(episodes_pre_train, scores_pre_train, facecolor='magenta', alpha=0.15,
                label = 'Score before training')  # plots the agent score
    ax3.plot(episodes_pre_train, mean_score_pre_train, linestyle = '-', color = 'cyan', lw=0.4,
             label = 'Mean score before training')  # plots the mean score of the last 100 eposides

    # This color doesnt matter if pre or pos training started since it only shows the goal
    ax3.plot(episodes, desired_score, 'g-', lw=0.4, label = 'Desired mean score')  # plots the desired score in each env

    ax3.legend(loc='best', fontsize = 'small')

    ax3.title.set_text('Episode Score')

    # Makes sure theres no overlap
    plt.tight_layout()

    if not os.path.isdir("./save_graph/"):
        os.makedirs("./save_graph/")

    fig.savefig("./save_graph/" + str(learning_rate) + desired_env + "_DDQN18.png")
    plt.close(fig)

def zero_to_nan(values):
    #Replace every 0 with 'nan' and return a copy
    return [float('nan') if x==0 else x for x in values]