% Generates plots that show the relationship between the distance of a cell
% from the centre of the aperture (as found by the flat-field correction
% script) and the mean fluorescence intensity (after flat-field correction)
% within the time series.

% Script is specific to the 2020-02-07 experiment (cExperiment '002'), in
% which flat-field correction has been attempted.

% Note: check if there are extra annotations via deleteannots.m first --
% will fix it so that I don't have to do this every time when I know what's
% going on.

% Ideally I'd like to work on the annotations (after removing pillars it
% thinks are cells), but annotations don't seem to store the x-y spatial
% position of traps, which I need.  There are only 10 pillars identified as
% cells though.

odb = SystemPreferences.omerodb;
odb.login;
cExperiment = odb.loadcExperiment('Arin_Batgirl_2020_Feb_07_flavin_pipeline_test_03', '002');
cellInf = cExperiment.cellInf(1);

% Calculates mean, min, max flavin fluorescence for each cell
y = cellInf.mean - cellInf.imBackground; % subtract background
means = mean(full(cellInf.mean) - full(cellInf.imBackground), 2);
mins = min(full(cellInf.mean) - full(cellInf.imBackground), [], 2); %
maxs = max(full(cellInf.mean) - full(cellInf.imBackground), [], 2);
  % different syntax for these for no good reason... why, MATLAB??

% Creates table that shows:
%     position number --
%     trap number     -- these two useful to identify 'good' timeseries later
%     mean y-value
%     min y-value --
%     max y-value -- used to define error bars in scatter plot (just to see what happens)
%     x position of trap
%     y position of trap
% for all cells
T_all = [];

% Goes through the 9 positions that have fluorescence and gets various
% attributes to put in T_all.
for kk = 4:12

    % defining things
    posIndex = kk;
    timepoint = 1;
    
    % pulls positions of traps for current position
    cTimelapse = cExperiment.returnTimelapse(posIndex);
    
    % stores x/y positions of where the traps are and find distances from
    % aperture centre
    xcenters = [];
    ycenters = [];
    aperture_centre = [532.0624, 604.5111]; % hard-coded, defined by aperture script
    for ii = 1:size(cTimelapse.cTimepoint(timepoint).trapLocations,2)
        xcenters = [xcenters cTimelapse.cTimepoint(timepoint).trapLocations(ii).xcenter];
        ycenters = [ycenters cTimelapse.cTimepoint(timepoint).trapLocations(ii).ycenter];
        distcenters = sqrt((xcenters - aperture_centre(1)).^2 + (ycenters - aperture_centre(2)).^2);
    end
    
    % calculates stuff for current position to put in T_all
    cellsOfInterest = cellInf.posNum == posIndex;
      % restrict cells of interest to position as defined -- this defines a logical
    trapNum = cellInf.trapNum(cellsOfInterest)'; % gets trap numbers
    xposition = [];
    yposition = [];
    distfromcentre = [];
    for jj = 1:sum(cellsOfInterest)
        distfromcentre = [distfromcentre distcenters(trapNum(jj))];
    end
    T = table(repmat(60*floor((kk -1)/3), sum(cellsOfInterest), 1),...
        repmat(kk, sum(cellsOfInterest), 1),...
        trapNum,...
        means(cellsOfInterest),...
        mins(cellsOfInterest),...
        maxs(cellsOfInterest),...
        distfromcentre');
    T.Properties.VariableNames = {'exposure',...
        'posNum',...
        'trapNum',...
        'TimeSeriesMean',...
        'TimeSeriesMin',...
        'TimeSeriesMax',...
        'distfromcentre'}; % end table creation
    T_all = [T_all; T];
end

T_all(any(ismissing(T_all),2),:) = []; 
  % removes NaNs (for some reason MATLAB requires a different function to
  % find NaNs for tables as opposed to matrices)
%%
% statistics and plotting
figure;
for ll = unique(T_all.exposure)' % goes through each exposure value
    % statistics
    ee = ll; % MATLAB is pretty stupid when it comes to for loops
    entriesOfInterest = T_all.exposure == ee;
    distfromcentre = T_all(entriesOfInterest,:).distfromcentre;
    TimeSeriesMean = T_all(entriesOfInterest,:).TimeSeriesMean;
      % selects T_all rows that has current exposure value
    [R, P] = corrcoef([distfromcentre, TimeSeriesMean]);
      % calculates:
      % matrix of correlation coefficients - R
      % matrix of p-values for testing the hypothesis that there is no
      % relationship between the observed phenomena - P
    
    % plotting
    subplot(3,1,ee/60);
    % scatter
    scatter(distfromcentre, TimeSeriesMean/ee, 'filled');
    % defining stuff for range bars
    TimeSeriesMin = T_all(entriesOfInterest,:).TimeSeriesMin;
    yneg = TimeSeriesMean - TimeSeriesMin;
    TimeSeriesMax = T_all(entriesOfInterest,:).TimeSeriesMax;
    ypos = TimeSeriesMax - TimeSeriesMean;
    hold on;
    %errorbar(distfromcentre, TimeSeriesMean/ee, yneg/ee, ypos/ee, 'o');
    % labelling stuff
    title({['Exposure ', num2str(ee), ' ms'],...
        ['correlation = ', num2str(R(1,2)),...
        ', p-value = ', num2str(P(1,2)),...
        ', N = ', num2str(sum(entriesOfInterest))]});
    xlim([0 500]);
    xlabel('Cell distance from centre of aperture (pixels)');
    ylim([0 0.02]);
    ylabel({'Average flavin fluorescence';...
        'of time series';...
        'normalised by exposure time (AU/min)'});
    hold off
end