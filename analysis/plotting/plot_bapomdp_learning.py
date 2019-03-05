""" this module plots the performance of a learning algorithm as a function of number of episodes"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np

MEAN_RETURN_COLUMN = 0
STDER_RETURN_COLUMN = 3

MARKERS = ['o', 'v', 's', 'p', '^', 'D', 'P', 'X']
MARK_COUNT = 0
MARKERS_PER_LINES = 5
MARK_AMOUNT = 0


def main():
    """ main """
    print("Processing a set of bapomdp.res files and plot lines out of them "\
            "given the x values and label (add --with-stder if you would like "\
            "to plot the standard error)\n")

    if not len(sys.argv) > 2:
        print("Requires 3 arguments:\n(1) file listing the bapomdp.res files\
                \n(2) label for each bapomdp.res file\n\
                with optional 3th argument --with-stder")
        exit(0)


    if len(sys.argv) == 4:
        if sys.argv[3] == "--with-stder":
            print_stder = True
        else:
            print("The only optional 5th argument is --with-stder")
            exit(0)
    else:
        print_stder = False

    arguments = {
            'input_file': sys.argv[1],
            'labels': sys.argv[2],
            }

    # load variables
    labels, bapomdp_results, stder = load_files(arguments)

    # we want each line to have the same (len(x)) length
    if len(bapomdp_results) != len(labels):
        sys.exit("Please enter as many bapomdp.res files (" + \
                str(len(bapomdp_results)) + ") as labels (" + \
                str(len(labels)) + ")")

        global MARK_AMOUNT
    MARK_AMOUNT = len(bapomdp_results)

    lengths = np.array([len(r) for r in bapomdp_results])
    if not (lengths == lengths[0]).all():
        sys.exit('not all files contain the same amount of values:' + str(lengths))

    num_episodes = lengths[0]

    for i, res in enumerate(bapomdp_results):
        if print_stder:
            plot_with_stder(range(num_episodes), res, stder[i], labels[i])
        else:
            plot_line(range(num_episodes), res, labels[i])

    plt.xlabel('# episodes')
    plt.ylabel('return per episode')

    plt.xlim(0, num_episodes)
    plt.title('place holder')
    plt.tight_layout()

    plt.legend(loc='lower right')
    plt.show()

def load_files(args): # {(int) 'n_episodes', (file_path) 'labels', (file_path) 'input_file'}
    """ loads in files and values from options """

    labels = open(args['labels']).read().splitlines()
    input_file = args['input_file']

    # for input_file in args['input_files']:
    basepath = "/".join(os.path.abspath(input_file).split('/')[:-1]) + "/"
    data = []
    stder = []
    for file_name in open(input_file).read().splitlines(): # add result from file
        data.append(np.loadtxt(basepath + file_name, delimiter=",")[:, MEAN_RETURN_COLUMN])
        stder.append(np.loadtxt(basepath + file_name, delimiter=",")[:, STDER_RETURN_COLUMN])

    return (labels, data, stder)

def plot_line(eps, data, label): # {float array, float array, string}
    """ plots a single line of data as a function of eps with label"""
    global MARK_COUNT
    plt.plot(eps, data, label=label, marker=MARKERS[MARK_COUNT], \
            markevery=( \
            int(((MARK_COUNT+1) * len(data)) / (MARK_AMOUNT * MARKERS_PER_LINES)), \
            (int(len(data)/MARKERS_PER_LINES))\
            ), ms=7, \
            markeredgecolor='k', \
            linewidth=.8, alpha=.9)
    MARK_COUNT += 1

def plot_with_stder(eps, data, stder, label): # {float array, float array, float array, string}
    """ plots a single line with error margin of data as a function of eps with label"""
    plot_line(eps, data, label)
    plt.fill_between(eps, data-2*stder, data+2*stder, alpha=.5)

if __name__ == '__main__':
    sys.exit(main())
