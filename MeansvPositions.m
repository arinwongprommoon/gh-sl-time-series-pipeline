% This script generates plots that show the average fluorescence intensity
% within each time series for all cells in an experiment, displayed
% according to the x-y spatial position within the device. Takes about a
% minute... there's obviously a more efficient way to do it.

% Script is specific to the 2020-02-07 experiment, but can be easily
% adapted to other experiments because it's very general.

% Top row: positions in which the exposure in the flavin channel is 60 ms
% Middle row: ... 120 ms
% Bottom row: ... 180 ms

odb = SystemPreferences.omerodb;
odb.login;
cExperiment = odb.loadcExperiment('Arin_Batgirl_2020_Feb_07_flavin_pipeline_test_03', '002');

% wrapping the entire damn thing into a big for loop so I can look at all 9
% positions that have fluorescence
figure;
for kk = 4:12

    % defining things
    posIndex = kk;
    timepoint = 1;
    
    % pulls positions of traps
    cTimelapse = cExperiment.returnTimelapse(posIndex);
    
    % stores x/y positions of where the traps are
    xcenters = [];
    ycenters = [];
    for ii = 1:size(cTimelapse.cTimepoint(timepoint).trapLocations,2)
        xcenters = [xcenters cTimelapse.cTimepoint(timepoint).trapLocations(ii).xcenter];
        ycenters = [ycenters cTimelapse.cTimepoint(timepoint).trapLocations(ii).ycenter];
    end
    
    % creates table that shows:
    %     trap number
    %     mean y-value
    %     x position of trap
    %     y position of trap
    % Note: I'm creating variables on the fly that are essentially the same
    % name as dot-indexed variables in a previous (used to by 100%
    % functional) version of the code because MATLAB suddenly decides to be
    % MATLAB for some reason. Why the hell do I put up with this?
    cExperiment_cellInf_posNum = cExperiment.cellInf.posNum;
    cellsOfInterest = cExperiment_cellInf_posNum == posIndex; % restrict cells of interest to position as defined -- this defines a logical
    cExperiment_cellInf_trapNum = cExperiment.cellInf.trapNum;
    trapNum = cExperiment_cellInf_trapNum(cellsOfInterest)';
    cExperiment_cellInf_mean = cExperiment.cellInf.mean;
    means = mean(full(cExperiment_cellInf_mean), 2); % gets mean y-values for each timeseries
    xposition = [];
    yposition = [];
    for jj = 1:sum(cellsOfInterest)
        xposition = [xposition xcenters(trapNum(jj))];
        yposition = [yposition ycenters(trapNum(jj))];
    end
    T = table(cExperiment_cellInf_trapNum(cellsOfInterest)', means(cellsOfInterest), xposition', yposition');
    T.Properties.VariableNames = {'trapNum', 'TimeSeriesMean', 'xposition', 'yposition'};
    
    % spatial representation of where the traps are, coloured by the mean
    % y-values
    subplot(3,3,kk-3);
    scatter(T.xposition, T.yposition, 50, T.TimeSeriesMean, 'filled');
    xlim([0 1200]);
    ylim([0 1200]);

end