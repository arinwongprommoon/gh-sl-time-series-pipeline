% NOTE TO SELF: VERY IMPORTANT -- REMOVE UNFLATTERING COMMENTS BEFORE
% PASSING THE FILE ON TO ANYBODY ELSE!!!!

% RAMON GRIMA (email, 2020-09-18):
% Note that i = 110:110 means that the power spectrum of the 110th cell is to 
% be calculated (since we doing spectra for each cell in a population).  
%% Jesus fucking Christ I hate reading other people's code
% files = dir('C:\Users\Chen Jia\Desktop\Oscillation Data\Analysis_MC4100_25C\MC4100_25C\*.txt'); flag = 25;
% files = dir('C:\Users\Chen Jia\Desktop\Oscillation Data\Analysis_MC4100_27C\MC4100_27C\*.txt'); flag = 27;
files = dir('C:\Users\Chen Jia\Desktop\Oscillation Data\Analysis_MC4100_37C\MC4100_37C\*.txt'); flag = 37;
numlineage = length(files);         % get the number of files
maxtime = 6000;
data = zeros(numlineage,maxtime);   % save gene expression data
div = zeros(numlineage,maxtime);    % save cell division data
numrow = zeros(1,numlineage);
for i = 1:numlineage
    filefolder = files(i).folder;   % get the names of folders
    filename = files(i).name;       % get the names of files
    temp = readmatrix(fullfile(filefolder,filename));    % read the data in this position
    [numrow(i),~] = size(temp);
    data(i,1:numrow(i)) = temp(:,4)'/100; % OH MY FUCKING GOD WHYYYYYYY JESUS FUCKING CHRIST WHAT THE F-U-C-K WENT INTO THIS BULLSHIT????
    div(i,1:numrow(i)) = temp(:,2)';
    maxtime = min([numrow(i),maxtime]); % what... the fuck... is this SHIT???? wait so you're telling me that you don't know ANY functions that tell you the length of any array?! Jesus fucking christ go jump off a cliff.
end

%             ,--.
%            /-__ \
%           |\\\   |
%           |\\\\/|D
%           = \  //
%       __=--`-\_\--,__
%      /#######\   \###`\
%     /         \   \|  |
%    /   /|      \   \  |
%   /   / |       \     |
%  /   /  |        \    /

%% WHY are you hard-coding the number of lineages rather than making a good data table???
% power spectrum computation
% 65 lineages for 25C
% 54 lineages for 27C
% 160 lineages for 37C
countI = 0; countIII = 0; % what the fuck is this for?
if flag == 25
    maxnum = 65; mid = 68; upper = 87; % what the fuck are those?
elseif flag == 27
    maxnum = 54; mid = 54; upper = 64;
else
    maxnum = 160; mid = 33; upper = 39;
end
%% christ all the fucking zeros is this C or something?

% defining my own time series: a noisy sinusoid
time_axis = 0:359;
%reading_axis = 2 * sin(2*pi*0.01*time_axis);
reading_axis_raw = readmatrix('test_ar.csv');
%reading_axis_raw = readmatrix('flavin72.csv');
reading_axis = reading_axis_raw';

maxnum = 360;
numrow = [360];
data = reading_axis;

% jesus christ learn how to name variables i swear to god
% also YOU DON'T NEED HALF THESE VARIABLES
itv = 1e-4;
freq = 0:itv:0.1;
len = length(freq);
spec = zeros(maxnum,len);
specp = zeros(maxnum,len);
specnp = zeros(maxnum,len);
wid = zeros(1,maxnum);
widp = zeros(1,maxnum);
widnp = zeros(1,maxnum);

i = 1;

% calculate the correlation function
numlineage = numrow(i); time = 1:numlineage; % W-H-A-T justifies this roundabout way of defining unnecessary variables?????
time = 1:numlineage;
x = data(i,time);
%y = div(i,time);
num = floor(3*sqrt(numlineage));
mu = mean(x);
gamma = zeros(1,num+1);
% I just spent 45 minutes making sure that there's no OBOE.
% FUCK MATLAB AND STARTING ARRAYS AT ZERO. TIME I WILL NEVER GET BACK
for j = 1:num+1
    for k = 1:numlineage-j+1
        gamma(j) = gamma(j)+(x(k)-mu)*(x(k+j-1)-mu)/numlineage;
    end
end

% calculate the power spectrum
AIC = zeros(1,num); BIC = zeros(1,num);
%for p = 1:num
for p = 56:56
    R = zeros(p);
    for j = 1:p
        for k = 1:p
            R(j,k) = gamma(abs(k-j)+1);
        end
    end
    v = zeros(p,1);
    for j = 1:p
        v(j) = gamma(j+1);
    end
    phi = -R\v; % note: there shouldn't be a minus sign. It's only there because the idiot decided to increment 'theta' below... rather than decrementing
    theta = gamma(1);
    for j = 1:p
        theta = theta+phi(j)*gamma(j+1); % erm... you can actually SUBTRACT the stuff on the right.  Fucking dumbass
    end
    AIC(p) = log(theta)+2*p/numlineage;
    %BIC(p) = log(theta)+p/numlineage*log(numlineage);
end

%%
% i'm not sure what these stupid variables are for, but it seems like
% they're related to the length of the time series.  AND THEY'RE
% HARD-CODED. jfc
%p = find(AIC(mid:upper)==min(AIC(mid:upper)))+mid-1;
% aaaaaand you're going through the damn thing again. functions exist
% OH WAIT I JUST REALISED THAT MATLAB DOESN'T LIKE FUNCTIONS dear gOD

p = 14;

R = zeros(p);
for j = 1:p
    for k = 1:p
        R(j,k) = gamma(abs(k-j)+1);
    end
end
v = zeros(p,1);
for j = 1:p
    v(j) = gamma(j+1);
end
phi = -R\v;
theta = gamma(1);
for j = 1:p
    theta = theta+phi(j)*gamma(j+1);
end
for j = 1:len
    temp = 1;
    for k = 1:p
        temp = temp+phi(k)*exp(-1i*2*pi*k*freq(j)) % what about the minus sign when you defined phi then...
    end
    specp(i,j) = specp(i,j)+theta/2/pi/abs(temp)^2; % AAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHH DON'T PUT YOUR DIVIDE SYMBOLS LIKE THIS!!!
    % also why are you.... adding... WAIT OH MY FUCKING GOD THEY'RE
    % JUST ADDING STUFF TO A Z-E-R-O.  AHHHH the idIOCy
end
specp(i,:) = specp(i,:)/specp(i,1); % normalise by the first value, so it seems
figure; plot(freq,specp(i,:),'b');   % plot the power spectrum
