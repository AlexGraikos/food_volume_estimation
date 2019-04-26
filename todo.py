#!/usr/bin/env python
from __future__ import print_function

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

todo = ['Implement depth-predicting model and run tests on it',
        'Check encoder->decoder skip connections',
        'Validate that python3 port is working properly (check divisions -> they are now altered)'
        'Clean up code!',
        'Add possibly other preprocessing parameters to dataframe and set creation utils in data_utils.py',
        'Run loads of tests']

done = ['Model can be trained, no actual tests ran',
        'ResNet18 instead of DenseNet used to reduce computational and memory costs',
        'Code in model and data_utils made prettier using cmd arguments',
        'Ported everything to python3']

        
        
title_format = bcolors.HEADER + bcolors.BOLD
todo_format = bcolors.OKGREEN + bcolors.UNDERLINE
task_format = bcolors.BOLD
endc = bcolors.ENDC
print(title_format + 'Food volume estimation project (April 2018-)' + endc) 
print('    -->', todo_format + 'TODO List' + endc)
for task in todo:
    print('        ++', task_format + task + endc)
print('    -->', todo_format + 'DONE List' + endc)
for task in done:
    print('        ++', task_format + task + endc)
