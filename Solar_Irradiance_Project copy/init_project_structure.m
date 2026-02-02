clear; clc; close all;

PROJECT_ROOT = pwd;

DIR.data_raw    = fullfile(PROJECT_ROOT,'data','raw');
DIR.results     = fullfile(PROJECT_ROOT,'results');
DIR.tables      = fullfile(DIR.results,'tables');
DIR.predictions = fullfile(DIR.results,'predictions');
DIR.figures     = fullfile(DIR.results,'figures');
DIR.logs        = fullfile(PROJECT_ROOT,'logs');

dirs = struct2cell(DIR);
for i = 1:length(dirs)
    if ~exist(dirs{i},'dir')
        mkdir(dirs{i});
    end
end

diary(fullfile(DIR.logs,'Solar_Irradiance_CommandLog.txt'));
diary on
tic;