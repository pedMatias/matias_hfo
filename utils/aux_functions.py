import matplotlib.pyplot as plt
import numpy as np


def plot_learning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def q_table_variation(old_q_table: np.ndarray, new_q_table: np.ndarray):
    diff_values = []
    for old_q, new_q in zip(old_q_table, new_q_table):
        diff = abs(new_q - old_q)
        diff_values.append(np.linalg.norm(diff))
    return sum(diff_values) / len(diff_values)


def get_mean_value_list_by_range(l: list, rang: int = 10):
    mean_list = []
    for idx, ele in enumerate(l):
        ir = idx - rang if idx > rang else 0
        fr = idx + rang if len(l) > idx + rang else len(l) - 1
        list_part = l[ir:fr]
        mean_list.append(sum(list_part)/len(list_part))
    return mean_list
