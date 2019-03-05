import sys
import os

import numpy as np

sys.setrecursionlimit(10000)

def main():
    if not len(sys.argv) > 2:
        print("Requires at least two files to merge")
        exit(0)

    # read in files
    input_files = []
    for i in range(1, len(sys.argv)):
        input_files.append(sys.argv[i]);

    file_statistics, dur = extract_statistics(input_files)

    # combine statistics
    combined_stats = combine_var_and_mean(file_statistics)
    dur            = np.mean(dur,0)
    stder          = np.sqrt(combined_stats['var'] / combined_stats['n'])

    print_stats(combined_stats, dur, stder)

def extract_statistics(input_files):
    # input_files = ['path a', 'path b']
    return_mean_index = 0;
    return_var_index = 1; 
    return_count_index = 2;
    duration_mean_index = 4;# 3 == stder

    statistics = []
    dur = []

    for f in input_files: 
        content = np.loadtxt(f, delimiter=",")
        if (len(content.shape) == 1): # planning results
            statistics.append( {
                'mu'  : content[return_mean_index],
                'var' : content[return_var_index],
                'n'   : content[return_count_index],
                        })

            dur.append(content[duration_mean_index])

        else: # bapomdp file
            assert(len(content.shape) == 2)
            statistics.append( {
                'mu'  : content[:,return_mean_index],
                'var' : content[:,return_var_index],
                'n'   : content[:,return_count_index],
                        })

            dur.append(content[:,duration_mean_index])

    return statistics, np.array(dur)

def combine_var_and_mean(statistics):
    # statistics = [{mu,var,n} ... ]
    # out: combined {mu, var, n} 

    stat1 = statistics[0]
    # recursion here
    stat2 = statistics[1] if (len(statistics) == 2) else combine_var_and_mean(statistics[1:])

    n = stat1['n']+stat2['n'] 
    mu = (stat1['mu']*stat1['n'] + stat2['mu']*stat2['n']) / n 

    var = real2sample(
            (
                stat1['n']*(sample2real(stat1['var'],stat1['n'])+stat1['mu']**2) + stat2['n']*(sample2real(stat2['var'],stat2['n']) + stat2['mu']**2) 
            )  /
            n - mu**2,
            n)

    return {'mu':mu,'var':var,'n':n}

def print_stats(combined_stats, dur, stder):
    # input option 1 (planning): [{'mu','var','n'}...] float float
    # input option 2 (bapomdp) : [{'mu','var','n'}...] [float...] [float...] 
    print("# version 1:\n# return mean, return var, return count, return stder, step duration mean")

    if combined_stats['mu'].shape == (): # planning results
            print(str(combined_stats['mu']) + ", " + str(combined_stats['var']) + ", " + str(combined_stats['n']) + ", "  + str(stder) + ", " + str(dur))

    else: # bapomdp results
        for i in range(len(combined_stats['mu'])):
            print(str(combined_stats['mu'][i]) + ", " + str(combined_stats['var'][i]) + ", " + str(combined_stats['n'][i]) + ", "  + str(stder[i]) + ", " + str(dur[i]))

##### transformations between real (biased) and sample (unbiased) variance
def sample2real(v,n):
    return ((n-1)/n) * v

def real2sample(v,n):
    return (n/(n-1)) * v

if __name__ == '__main__':
    sys.exit(main());
