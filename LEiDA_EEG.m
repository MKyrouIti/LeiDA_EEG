load('All_data.mat'), load sensor_names,
NAMES = fieldnames(All_data),
Fs=300; 

%% Part I: reading the STs
STs=[]; valence=[]; arousal=[];liking=[]; Labels_stim=[]; IDs=[];Bas_STs1=[];

for ii=1:42
Subject_STs=eval(char(strcat('All_data.',NAMES(ii),'.trials')));Subject_Baselines=eval(char(strcat('All_data.',NAMES(ii),'.baselines')));
Subject_labels=table2array(eval(char(strcat('All_data.',NAMES(ii),'.labels'))));
Stim_labels_visual=eval(char(strcat('All_data.',NAMES(ii),'.trial_permutation.visual')));

% Ave Re-Reference
for i_trial=1:48, 
    ST_DATA=Subject_STs(:,:,i_trial); 
    ST_DATA=[ST_DATA-mean(ST_DATA)];Subject_STs(:,:,i_trial)=ST_DATA;
end

% first 16 trials out of 48 
STs=cat(3,STs,Subject_STs(:,1:end,1:16)); 
liking=[liking;Subject_labels(1:16,3)]; 
arousal=[arousal;Subject_labels(1:16,2)];
valence=[valence;Subject_labels(1:16,1)];
Labels_stim=[Labels_stim;Stim_labels_visual'];
IDs=[IDs;ii*ones(16,1)];

% BandLimiting & Ave Re-Reference for Baselines
Bas_DATA=Subject_Baselines(:,:,1);Bas_DATA=[Bas_DATA-mean(Bas_DATA)];Subject_Baseline1=Bas_DATA; 
Bas_STs1(:,:,ii)=Subject_Baseline1;
end 

time=[1:size(STs,2)]*(1/Fs);
 
%%  PART II :  deriving average Histograms, prototypical of each state: "Like"  and "Dislike"
for ii=1:16; 
    Range_liking(ii)=iqr(liking(find(Labels_stim==ii))); 
end

[sorted,slist]=sort(Range_liking,'descend'); % sorted list of paintings in terms of "usefulness" in defining 'Like'-'DisLike'

LL=Labels_stim==[slist(1:12)]; list_for_paint=find(sum(LL,2)); %select paintings 

Ths=prctile(liking(list_for_paint),[25,75]); 
Th_low=Ths(1); Th_high=Ths(2); % the Low and High level threshold for defining groups of trials

list_high=list_for_paint(find(liking(list_for_paint)>=Th_high)); % the list of 'Like' trials
list_low=list_for_paint(find(liking(list_for_paint)<=Th_low)); % the list of 'DisLike' trials

keep_idx = [list_low; list_high]; % trials' IDs
SubID_sel=IDs(keep_idx); % subjects' IDs
Bas_STs1=Bas_STs1(:,:,unique(SubID_sel)); 

paint_STs=STs(:,:,[list_low;list_high]); % the 3D array of selected multichannel trials
paint_labels=[zeros(numel(list_low),1);ones(numel(list_high),1)]; % the associated binary labels

[Nsensors,Ntime,Ntrials]=size(paint_STs);

%%
Fband=[1 4]; % delta band 
K_cluster_numbers=10; Plevel=[]; WWlevel=[]; 

[b,a]=butter(3,[Fband(1),Fband(2)]/(Fs/2));

ALL_LE=[];
for i_trial=1:Ntrials
DATA=paint_STs(:,1+Fs:end,i_trial);
fDATA=filtfilt(b,a,DATA')';
[Nsensors,Ntime]=size(fDATA);PHASES=angle(hilbert(fDATA'))'; % Nsensors x Ntime

Leading_Eig=[];
    for i_t=1:Ntime
    fi=PHASES(:,i_t); iFC=cos(repmat(fi,1,19)-repmat(fi',19,1));
    [V1,~]=eigs(iFC,1);
     if mean(V1>0)>.5, V1=-V1;
     elseif mean(V1>0)==.5 && sum(V1(V1>0))>-sum(V1(V1<0)),V1=-V1;
     end
      Leading_Eig(:,i_t)=V1;
    end 
ALL_LE=[ALL_LE,Leading_Eig];
end

size(ALL_LE) %  Timeseries of  leading Eigenvectors : 19 sensors x (Ntrials x Ntime) 

% perform C-means clustering
[IDX, Centers, SUMD, D]=kmeans(ALL_LE',K_cluster_numbers,'Replicates',10,'MaxIter',1000,'Display','off','Options',statset('UseParallel',1));

% from timeSeries of Cluster Memeberships to signle-subject's Patterns of Probability of occurence of each prototypical Leading Eignevector  
GroupsTS=reshape(IDX,Ntrials,[]);ProbPatterns=[]; 

for ii=1:Ntrials
T1=(tabulate(GroupsTS(ii,:))); 
ProbPatterns(ii,:)=T1(:,3)';
end

    %% Baseline: assign to centroids, make histograms

    ALL_LE_Bas = [];
    for i_sub = 1:size(Bas_STs1,3)
        DATA_Bas  = Bas_STs1(:, :, i_sub);
        fDATA_Bas = filtfilt(b,a, DATA_Bas')';
        PHASES = angle(hilbert(fDATA_Bas'))';

        Leading_Eig = zeros(size(fDATA_Bas,1), size(fDATA_Bas,2));
        for i_t = 1:size(fDATA_Bas,2)
            fi = PHASES(:,i_t);iFC = cos(repmat(fi,1,19) - repmat(fi',19,1));
            [V1,~] = eigs(iFC,1);
            if mean(V1>0) > .5
                V1 = -V1;
            elseif mean(V1>0) == .5 && sum(V1(V1>0)) > -sum(V1(V1<0))
                V1 = -V1;
            end
            Leading_Eig(:,i_t) = V1;
        end
        ALL_LE_Bas = [ALL_LE_Bas, Leading_Eig]; 
    end

    
    [IDX_Bas, D_Bas] = knnsearch(Centers, ALL_LE_Bas'); % assign to centroids
    GroupsTS_Bas = reshape(IDX_Bas, size(Bas_STs1,3), []);
    ProbPatterns_Bas = zeros(size(Bas_STs1,3), K_cluster_numbers);

    for ii = 1:size(Bas_STs1,3)
        T1_Bas = tabulate(GroupsTS_Bas(ii,:));ProbPatterns_Bas(ii,:) = T1_Bas(:,3)'; 
    end



%% Probabilities of Occurrence and Transition matrices
out = perm_test(GroupsTS, paint_labels, 10, 10000,  true, GroupsTS_Bas);
