trainingtracker
===============

training analysis for behavioral experiments -- version 1b

version 1b:
Specify input path (dir with a subfolder per animal) in command line.

EX: python analysis_script.py 'path/to/dir'

version 1a:
Each .py file should be in the same directory as an "input" folder.

The "input" folder should contain a sub-folder, with subject's name,
such that animals A1, A2, etc. training with a particular protocol each 
has a folder with its raw .mwk datafile.

EX: "analysis" folder contains .py analysis scripts with "input" folder containing subfolders (A1, A2, etc.) for each animal. This is where that animal's .mwk data files go. Can also have an "output" folder with corresponding subfolders for each animal's summary stats.