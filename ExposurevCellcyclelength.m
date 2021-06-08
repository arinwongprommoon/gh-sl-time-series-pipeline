%% Choose data, cells, and load data
% loads cExperiment file
odb = SystemPreferences.omerodb;
odb.login;
%expt = odb.loadcExperiment('Arin_Batgirl_2019_Nov_29_flavin_pipeline_test_00', '001');
expt = odb.loadcExperiment('Arin_Batgirl_2020_Feb_07_flavin_pipeline_test_03', '002');
myinf = expt.cellInf(1);
%% Pool positions according to experimental groups (2020-01-16 experiment)
for ii = 1:size(myinf.posNum,2)
    if myinf.posNum(ii) == 5 | myinf.posNum(ii) == 6 % redundancy for readability
        myinf.posNum(ii) = 5
    elseif myinf.posNum(ii) == 7 | myinf.posNum(ii) == 8 | myinf.posNum(ii) == 9
        myinf.posNum(ii) = 6
    elseif myinf.posNum(ii) == 10 | myinf.posNum(ii) == 11 | myinf.posNum(ii) == 12
        myinf.posNum(ii) = 7
    elseif myinf.posNum(ii) == 13 | myinf.posNum(ii) == 14 | myinf.posNum(ii) == 15
        myinf.posNum(ii) = 8
    end
end
% Resulting groups:
% 1 - SC,   0 ms
% 2 - SC,  60 ms
% 3 - SC, 120 ms
% 4 - SC, 180 ms
% 5 - SM,   0 ms
% 6 - SM,  60 ms
% 7 - SM, 120 ms
% 8 - SM, 180 ms
%% Pool positions according to experimental groups (2020-01-31 and 2020-02-07 experiments)
for ii = 1:size(myinf.posNum,2)
    if myinf.posNum(ii) == 1 | myinf.posNum(ii) == 2 | myinf.posNum(ii) == 3
        myinf.posNum(ii) = 1
    elseif myinf.posNum(ii) == 4 | myinf.posNum(ii) == 5 | myinf.posNum(ii) == 6
        myinf.posNum(ii) = 2
    elseif myinf.posNum(ii) == 7 | myinf.posNum(ii) == 8 | myinf.posNum(ii) == 9
        myinf.posNum(ii) = 3
    elseif myinf.posNum(ii) == 10 | myinf.posNum(ii) == 11 | myinf.posNum(ii) == 12
        myinf.posNum(ii) = 4
    end
end
% Resulting groups:
% 1 -   0 ms
% 2 -  60 ms
% 3 - 120 ms
% 4 - 180 ms
%% Gets mean lengths of cell cycle in each experimental group (2020-02-07)
% Note from Arin of April 2020: Arin of January 2020 wrote this.  It's
% crap, but it works.  Don't fuck around with it; I've only made minor
% changes, and let's keep it that way.  Also it will be outdated soon anyway.
sampling_pd = 2.5;

posmeancyclelengths = [];
allmeancyclelengths = [];
positions = [];
cyclelengthsinpos = [];
h = [];

myinf_births = full([myinf.births]); % (cell,timepoint) to which BABY assigns a birth
myinf_posNum = myinf.posNum; % group numbers for all cells

% gets lengths of cell cycles for each experimental group
for ii = 1:length(unique(myinf.posNum)) % goes through each experimental group
    ll = size(cyclelengthsinpos,2);

    posbirths = myinf_births(myinf_posNum == ii,:); % (cell, timepoint) to which BABY assigns a birth
    
    % gets lengths of cell cycles
    for jj = 1:size(posbirths,1) % goes through the cells for which there are birth events
        if ~isnan(mean(diff(find(posbirths(jj,:))))) % if things are biologically valid
             % (fuck knows what it actually means)
            cyclelengthsinpos = [cyclelengthsinpos, diff(find(posbirths(jj,:)))]; % appends lengths of cell cycles
        end
    end
    
    kk = size(cyclelengthsinpos,2);
    positions = [positions, ii*ones(1,kk-ll)];
      % tops up 'positions' with the current position to match the length of 'cyclelengthsinpos'.
      % seriously, why the FUCK did i do this
end
%% Boxplots (2020-02-07)
% Generates boxplots to show how the cell cycle lengths are distributed for
% each experimental group
figure;
boxplot(sampling_pd*cyclelengthsinpos, 60*(positions - 1));
xlabel('Flavin channel exposure time (ms)');
ylabel('Cell cycle length (min)');
%% Two-sample KS tests -- compare to control
% The test indicates whether two vectors come from the same continuous
% distribution. 

% Here, I'm comparing the experimental groups (i.e. exposure
% != 0 ms) with the 'control' group (i.e. exposure = 0  ms). Code assumes
% that position 1 is the control group.

for ll = unique(positions)
    ll
    h = kstest2(cyclelengthsinpos(positions == 1), cyclelengthsinpos(positions == ll))
end
%% Two-sample KS tests -- each pair

% And here, I'm comparing all the pairs.
% this is a very MATLAB-ic way of dealing with it...
% also this isn't a pretty display, because i couldn't be bothered
combinations = nchoosek(unique(positions), 2);
for mm = 1:size(combinations, 1)
    printthis = combinations(mm,:)
    h = kstest2(cyclelengthsinpos(positions == combinations(mm,1)), cyclelengthsinpos(positions == combinations(mm,2)))
end
%% Exports cell cycle lengths (2020-01-31 and 2020-02-07)
% this is pretty bad coding and can cause confusion, but i'll keep it like
% this till i could be bothered to think of a better way to deal with it.
% some of the stuff is for the sake of consistency with previous files i
% had created manually
id = (1:size(positions,2))';
exposure = ((positions - 1) * 60)';
cyclelength = cyclelengthsinpos';
position = positions';
T = table(id, cyclelength, position, exposure);
writetable(T, 'cyclelengths.csv');