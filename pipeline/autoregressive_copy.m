% RAMON GRIMA (email, 2020-09-18):
% Note that i = 110:110 means that the power spectrum of the 110th cell is to 
% be calculated (since we doing spectra for each cell in a population).  

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
    data(i,1:numrow(i)) = temp(:,4)'/100;
    div(i,1:numrow(i)) = temp(:,2)';
    maxtime = min([numrow(i),maxtime]);
end

% power spectrum computation
% 65 lineages for 25C
% 54 lineages for 27C
% 160 lineages for 37C
countI = 0; countIII = 0;
if flag == 25
    maxnum = 65; mid = 68; upper = 87;
elseif flag == 27
    maxnum = 54; mid = 54; upper = 64;
else
    maxnum = 160; mid = 33; upper = 39;
end
itv = 1e-4;
freq = 0:itv:0.1;
len = length(freq);
spec = zeros(maxnum,len);
specp = zeros(maxnum,len);
specnp = zeros(maxnum,len);
wid = zeros(1,maxnum);
widp = zeros(1,maxnum);
widnp = zeros(1,maxnum);
for i = 110:110   % cell number
    % calculate the correlation function
    numlineage = numrow(i); time = 1:numlineage;
    x = data(i,time);
    y = div(i,time);
    num = floor(3*sqrt(numlineage));
    mu = mean(x);
    gamma = zeros(1,num+1);
    for j = 1:num+1
        for k = 1:numlineage-j+1
            gamma(j) = gamma(j)+(x(k)-mu)*(x(k+j-1)-mu)/numlineage;
        end
    end
    
    % calculate the power spectrum
    AIC = zeros(1,num); BIC = zeros(1,num);
    for p = 1:num
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
        AIC(p) = log(theta)+2*p/numlineage;
        BIC(p) = log(theta)+p/numlineage*log(numlineage);
    end
    p = find(AIC(mid:upper)==min(AIC(mid:upper)))+mid-1;
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
            temp = temp+phi(k)*exp(-1i*2*pi*k*freq(j));
        end
        specp(i,j) = specp(i,j)+theta/2/pi/abs(temp)^2;
    end
    specp(i,:) = specp(i,:)/specp(i,1);
    figure; plot(freq,specp(i,:),'b');   % plot the power spectrum
end
