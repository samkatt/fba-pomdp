# Analysis on results 

There are two types of scripts here, those that process result files in order
to merge them, and those used to plot results. All the (python) scripts should
output a very short usage description when called to figure out the arguments.

## Plotting

The results produced by the experiments in this project are fairly simple, so
plotting comes down to pointing out which files to show. 

### Plotting results from learning (bapomdp, fbapomdp) results

Assuming you have 2 result files (file1.res and file2.res) from running
experiments, prepare the following files (see analysis/plotting/example):

#### files.txt:
```
file1.res
file2.res
```

##### labels.txt:
```
file 1 results
file 2 results
```

Then plot the results as follows (simply provide the files to plot and their
labels):

``` console
python plot_bapomdp_learning.py files.txt labels.txt
```

### Plotting results from 'planner' experiments or compare end result learning experiments

Whereas the output of learning experiments cover multiple episodes, planning
experiments have no such concept of time. As a result, you want to control the
x-axis. Similarly, you may want to compare just end performance (after
learning) of multiple experiments on a specific x-axis. Exploiting the
command-line flexibility, this all is managed through files. To plot lines with
a specific x-axis and labels, run:

```
python plot_planning_results x_values.txt labels.txt first_line.txt second_line.txt
```

Assuming files:

#### x_values.txt: (the values on the x-axis)
```
1
2
3
4
5
```

#### labels: 
```
label of first line
label of second line
```

#### first_line.txt
```
file/of/result/for/x/is/1
file/of/result/for/x/is/2
file/of/result/for/x/is/3
file/of/result/for/x/is/4
file/of/result/for/x/is/5
```

#### first_line.txt
```
another-file/of/result/for/x/is/1
another-file/of/result/for/x/is/2
another-file/of/result/for/x/is/3
another-file/of/result/for/x/is/4
another-file/of/result/for/x/is/5
```

## Processing result files

Sometimes you run the same learning experiment multiple times and would like to
merge the results (see analysis/preprocess/example).

``` console
python merge_result_files.py 1.res 2.res > merged.res
```

# Analysis on results 

There are two types of scripts here, those that process result files in order
to merge them, and those used to plot results. All the (python) scripts should
output a very short usage description when called to figure out the arguments.

## Plotting

The results produced by the experiments in this project are fairly simple, so
plotting comes down to pointing out which files to show. 

### Plotting results from learning (bapomdp, fbapomdp) results

Assuming you have 2 result files (file1.res and file2.res) from running
experiments, prepare the following files (see analysis/plotting/example):

#### files.txt:
```
file1.res
file2.res
```

##### labels.txt:
```
file 1 results
file 2 results
```

Then plot the results as follows (simply provide the files to plot and their
labels):

``` console
python plot_bapomdp_learning.py files.txt labels.txt
```

### Plotting results from 'planner' experiments or compare end result learning experiments

Whereas the output of learning experiments cover multiple episodes, planning
experiments have no such concept of time. As a result, you want to control the
x-axis. Similarly, you may want to compare just end performance (after
learning) of multiple experiments on a specific x-axis. Exploiting the
command-line flexibility, this all is managed through files. To plot lines with
a specific x-axis and labels, run:

```
python x_values.txt labels.txt first_line.txt second_line.txt
```

Assuming files:

#### x_values.txt: (the values on the x-axis)
```
1
2
3
4
5
```

#### labels: 
```
label of first line
label of second line
```

#### first_line.txt
```
file/of/result/for/x/is/1
file/of/result/for/x/is/2
file/of/result/for/x/is/3
file/of/result/for/x/is/4
file/of/result/for/x/is/5
```

#### first_line.txt
```
another-file/of/result/for/x/is/1
another-file/of/result/for/x/is/2
another-file/of/result/for/x/is/3
another-file/of/result/for/x/is/4
another-file/of/result/for/x/is/5
```


## Processing result files

Sometimes you run the same learning experiment multiple times and would like to
merge the results (see analysis/preprocess/example).

``` console
python merge_result_files.py 1.res 2.res > merged.res
```
