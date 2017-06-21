% This script can be used to interactively inspect the results

addpath('/home/parapa/github/selam/benchmark/vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

trackers = tracker_load('bin');

workspace_browse(trackers, sequences, experiments);

