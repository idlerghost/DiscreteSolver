import numpy as np
import os
from collections import deque
from auxFuncs import multicolor_plotter

def playGame(env, agent, env_max_score, desired_env, learning_rate, max_episodes, state_size, 
            sol_score, step, action_size):
    # Variables for plotting
    scores, episodes, action_list, memory_list, desired_score, mean_score, train_done, inverse_train_done = \
        [], [], [], [], [], [], [], []
    last_100_scores = deque(maxlen= 100)
    last_500_scores = deque(maxlen= 500)

    # Variable for saving the model
    max_score = -999

    # If we are not training aka we are running, load the model
    if (not agent.train):
        print("Now we load the saved model")
        agent.load_model("./save_model/" + str(learning_rate) + desired_env + "_DDQN18.h5")

    # If we are training:
    for e in range(max_episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            # If render is set to true, show the user the env. 
            # This makes the process slower.
            if agent.render:
                env.render()

            step += 1

            agent.update_step(step)

            # Get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # If an action make the episode end, then give it penalty of -1
            # Score for this is max number of steps -1

            # The reward is saturated between -1 and 1
            if not done:
                reward = reward
            elif done and score >= (env_max_score - 1):
                reward += 500
            else:
                reward += -500
            reward = np.clip(reward, -1, 1)  

            # Save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # Every time step do the training if we are training
            if step >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # Resets the env for new episode
                env.reset()
                # Every episode update the target model to be same with model
                agent.update_target_model()

                # Append the values for the graph
                scores.append(score)
                last_100_scores.append(score)
                last_500_scores.append(score)
                episodes.append(e)

                desired_score.append(sol_score)
                mean_score.append(np.mean(last_100_scores))
                if step < agent.train_start:
                    train_done.append(0)
                    inverse_train_done.append(1)
                else:
                    train_done.append(1)
                    inverse_train_done.append(0)

                # Every episode, plot the play time
                multicolor_plotter(episodes, scores, desired_score, train_done, mean_score, learning_rate, desired_env, inverse_train_done)
                # plot_model(agent.model, to_file=('./' + desired_env + str(learning_rate) +'model.png'), show_shapes= True)

                # Tell the user how things are going
                print(" | ", desired_env, " | Episode:", e, " | Score:", round(score,2), "/", env_max_score, " | Step:",
                      (step), "/", agent.train_start, " | Epsilon:", round(agent.epsilon,3), " | Reward Given:", reward,
                      " | Mean Score:",  round(np.mean(last_100_scores),3), "/", sol_score,  " |", "Max Memory:",
                      agent.memory_size)

                # If the mean of scores of last 100 episodes is the solution score
                # Stop training ~ Commented for now to get more data
                if np.mean(last_100_scores) >= sol_score and agent.train:
                    return
                if abs(np.mean(last_500_scores)) <= abs(sol_score/3) and len(scores) >= agent.train_start * 1.25:
                    return
                if abs(np.mean(last_500_scores)) <= abs(sol_score/2) and len(scores) >= agent.train_start * 1.5:
                    return

                # Greedy DQN
                if (score >= max_score and agent.train):
                    print("Now we save the better model")
                    max_score = score
                    if not os.path.isdir("./save_model/"):
                        os.makedirs("./save_model/")

                    agent.save_model(
                        "./save_model/" + str(learning_rate) + desired_env + "_DDQN18.h5")