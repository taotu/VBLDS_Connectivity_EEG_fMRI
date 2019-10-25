function [data] = VB_MStep(data)

% Gaussian and Gamma Priors
a0 = data.init.a0;      % parameters for connectivity precision
b0 = data.init.b0;      % parameters for connectivity precision parameters
c0 = data.init.c0;      % parameters for hyperparameters alpha
d0 = data.init.d0;      % parameters for hyperparameters alpha

% Noise covariance initial values
Qx0 = data.init.Qx0;
Qy0 = data.init.Qy0;
Re0 = data.init.Re0;
v0 = data.init.v0;

% State updates from VB_EStep
mutT = data.mutT;
VtT = data.VtT;
Pt = data.Pt;
Ptt_1 = data.Ptt_1;

%######################EEG parameters############################
EEGDim = size(Qy0,1); 
sDim = size(mutT,1); 
m = data.m; 
K = size(m,1);


if (~data.posteriorMStepFlag)
    c_post = c0*ones((K+1)*sDim+1,sDim);
    d_post = d0*ones((K+1)*sDim+1,sDim);
    init_Qx = Qx0;
    init_Qy = Qy0;
    init_Qu = chol(init_Qy);    
    data.cost = 0;
else
    c_post = data.cPst;
    d_post = data.dPst;
    init_Qx = data.QxPst;
    init_Qu = data.QuPst;
end

% Set this flag to 1 so the next iteration starts using posterior estimates
data.posteriorMStepFlag = 1;

%######################Latent State Parameters############################
% Estimate state parameters eta and beta 
[etaMu_post,etaV_post,Sigma_post,a_post,b_post,etaVariance,ELBO1] = state_parameter_update(data,mutT,VtT,Pt,Ptt_1,a0,b0,c_post,d_post);

% Estimate state hyperparameters alpha 
[c_post,d_post,ELBO2] = state_hyperparameter_update(c0,d0,etaMu_post,Sigma_post,a_post,b_post);

%######################EEG Parameters############################
% update the noise covariance for EEG
[Qu_post,Qx_post,RePst,vPst,cost] = EEG_observation_parameter_update_wishart(data,mutT,VtT,Re0,v0,init_Qu,init_Qx);
ELBO3 = -size(mutT,2)/2*log(2*pi)-cost/2;

ELBO = ELBO1+ELBO2+ELBO3;

% Posterior updates
data.aPst = a_post;
data.bPst = b_post;
data.cPst = c_post;
data.dPst = d_post;
data.QuPst = Qu_post;


Qu_post_reshape = reshape(Qu_post,EEGDim,EEGDim);


data.QyPst = Qu_post_reshape'*Qu_post_reshape;
data.QxPst = Qx_post;


data.etaMuPst = etaMu_post;
data.etaVPst = etaV_post;
data.SigmaPst = Sigma_post;
data.etaVariance = etaVariance;

data.ELBO = ELBO;
data.Q0KLPst = data.mu0T*data.mu0T';
% Transfrom eta, beta and lambda into A,B,D,Q,R
[data] = VB_paramterReshape(data);

data.RePst = RePst;
data.vPst = vPst;
end

