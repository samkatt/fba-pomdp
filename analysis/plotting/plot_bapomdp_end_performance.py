import sys
import os

import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Process a set of files containing a list of bapomdp.res files and plots a line out of each of them given the x values and label\n")

    if (not len(sys.argv) > 4):
        print("Requires 3 arguments:\n(1) file containing x values\n(2) file containing labels and\n(3) a set of files where each file lists the files associated with one line")
        exit(0)

    input_files = []
    for i in range(3, len(sys.argv)):
        input_files.append(sys.argv[i]);

    arguments = {
            'x': sys.argv[1],
            'labels': sys.argv[2],
            'input_files': input_files
            }

    # load variables
    (y_array, x, labels) = load_files(arguments)

    # we want each line to have the same (len(x)) length
    for y in range(len(y_array)): 
        if len(y_array[y]) != len(x):
            sys.exit("Please make sure the # of files in input file " + str(i) + " is " + str(len(x)) + " (was " + str(len(y_array[y])) + ")")

    if len(y_array) != len(labels):
        sys.exit("The number of labels (" + str(len(labels) + ") does not match the number of lines (" + str(len(y_array))))

    for i in range(len(y_array)):
        plt.plot(x, y_array[i], label=labels[i])

    plt.xscale('log')
    plt.legend()
    plt.show()

def load_files(args): # {(file path) 'x', (file_path) 'labels', (arr file paths) 'input_files'}
    """ loads in files and values from options """

    x      = np.loadtxt(args['x'], delimiter=",")
    labels = open(args['labels']).read().splitlines()

    y_array = []
    for input_file in args['input_files']:

        basepath = "/".join(os.path.abspath(input_file).split('/')[:-1]) + "/"
        y = []
        for f in open(input_file).read().splitlines(): # add result from exp file 
            y.append(np.loadtxt(basepath + f, delimiter=",")[-1,0]) # store last return from the file

        y_array.append(y)

    return (y_array, x, labels)

if __name__ == '__main__':
    sys.exit(main());

