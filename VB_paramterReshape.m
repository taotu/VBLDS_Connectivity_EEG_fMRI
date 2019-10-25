function [data] = VB_paramterReshape(data)


if (~data.posteriorMStepFlag)
    a_post = data.init.a0;
    b_post = data.init.b0; 
    disp('Fail to run VB-M Step')
else
    a_post = data.aPst;
    b_post = data.bPst;
    QyPst = data.QyPst;
    QxPst = data.QxPst;
    etaMu_post = data.etaMuPst;
    etaVariance = data.etaVariance;
end

% Dimensions of model parameters
m = data.m;
K = size(m,1);      % number of modulatory input
sDim = size(data.mutT,1);       % dimension of latent variables


if (~data.posteriorEStepFlag)
    disp('Fail to run VB-E Step')
else
    % Extract the mean for A
    data.APst = etaMu_post(1:sDim,:)';
    % Extract the covariance matrix for A
    data.AVariancePst = etaVariance(1:sDim,:)';

    data.BPst = [];
    data.BVariancePst = [];

    for k = 1:K
         data.BPst = cat(3,data.BPst, etaMu_post(1+k*sDim:(k+1)*sDim,:)');
         data.BVariancePst = cat(3,data.BVariancePst, etaVariance(1+k*sDim:(k+1)*sDim,:)');
    end

    data.DPst = diag(etaMu_post((K+1)*sDim+1,:));
    data.DVariancePst = diag(etaVariance((K+1)*sDim+1,:));
    
    % State noise covariance matrix 
    data.QPst = diag(b_post./a_post);
    
    % EEG noise covariance matrix 
    % Reshape Qx to diagonal matrix
    Qx_reshape = zeros(size(data.L,2),1);
    sigma0G = zeros(size(data.L,2),1);
    for k = 1:sDim
        Qx_reshape(logical(data.G(:,k))) = QxPst(k)^2;
        sigma0G = sigma0G | data.G(:,k);
    end
    Qx_reshape(sigma0G) = QxPst(sDim+1)^2;
    
    Qx_reshape = diag(Qx_reshape);  
    data.RePst = QyPst+data.L*Qx_reshape*data.L';


    diag_R = diag(data.RePst);
    negElementsR = diag_R<0;

    diag_Q = diag(data.QPst);
    negElementsQ = diag_Q<0;
    if sum(negElementsR)>0 || sum(negElementsQ)>0
    data.flag = 1;
    end

end
