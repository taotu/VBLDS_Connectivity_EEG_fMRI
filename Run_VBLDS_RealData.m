clc;
clear;
close all;
%%
dataPath = '/Users/taotu/Documents/LabWork/SajdaLab/State_Space_EEG_fMRI/NIPS_Paper/Final/Data/';

subjects = {'FCH_15_M26_04292011', 'FCH_23_F28_06222011', 'FCH_18_F20_06132011', 'FCH_36_F26_08172011','FCH_31_M23_07252011', ...
    'FCH_22_M30_06212011', 'FCH_41_F21_08252011', 'FCH_29_F23_07202011','FCH_25_F20_07062011','FCH_37_M35_08182011'};

for subject = 1:length(subjects)
    
    for blockNum = 2;
    datafile = fullfile(dataPath,subjects{subject},['EEG_fMRI_Block' num2str(blockNum) '.mat']);
    load(datafile);
    
    % Set up parameters for VBLDS
    T = size(data.EEG,2);       % number of observations
    sDim = size(data.G,2);      % dimension of latent states
    
    % Normalize each row in the lead-field matrix
    L = data.L;
    A = zeros(size(L)); % normalize L by rows
    for k = 1:size(L,1)
        A(k,:) = L(k,:)/sqrt(sum(L(k,:).^2));
    end
    data.L = A;     % Normalized lead-field matrix

    data.m = data.m_category(:,1:T);        % modulatory input
    u = zeros(sDim,T);
    % Set up the external input
    u(1,:) = data.u(1,1:T)+data.u(2,1:T);
    u(2,:) =  data.u(3,1:T)+data.u(2,1:T);
    data.u = u;     % external input
    
    % Run VBLDS
    BDS = runVBLDS(data);
    
    % Save result files
    resultPath = '/Users/taotu/Documents/LabWork/SajdaLab/State_Space_EEG_fMRI/NIPS_Paper/Final/Results/';
    resultFile = fullfile(resultPath,[subject 'block' num2str(blockNum) '.mat']);
    save(resultFile,'BDS');
    clear BDS
    end
end
