function [data] = VB_EStep(data)

if (~data.posteriorEStepFlag)
    
    % State parameters initial values
    A = data.init.A0;
    B = data.init.B0;
    D = data.init.D0;
    Q = data.init.Q0;       % state covariance
    
    % EEG observation parameters initial values
    Re = data.init.Re0;     % EEG observation covariance
 
    % Q0 in the kalman filter
    Q0KL = data.init.Q0KL; 
    
    
else
    
    % State parameters from previous update
    A = data.APst;
    B = data.BPst;
    D = data.DPst;
    Q = data.QPst;
    
    % EEG observation parameters from previous update
    Re = data.RePst;
    % Q0 in the kalman filter
    Q0KL = data.Q0KLPst;
end

data.posteriorEStepFlag = 1;

% Known quantities  
u = data.u;     % external input
L = data.L;     % lead-field matrix
G = data.G;     % source roi membership matrix 
C = L*G;        % EEG emission matrix
yEEG = data.EEG;    % EEG observation

% if there is modulatory input
if isfield(data,'m')
    m = data.m;     % modulatory inputs, each row corresponds to one modulatory input
    
    % Initialization of kalman filter
    mu00 = (Q0KL*C')/(C*Q0KL*C'+Re)*randn(size(C,1),1);
    V00 = data.init.Q0KL;
    
    % Kalman-filtering
    [mut,mutt_1,Vt,Vtt_1,logLik] = KL_filtering_forward(yEEG,u,A,B,m,mu00,D,Q,V00,C,Re);

    % Kalman-smoothing
    [mutT,VtT,Pt,Ptt_1,mu0T] = KL_smoothing_backward(A,B,m,D,u,mut,Vt,Vtt_1,Vtt_1(:,:,1),mutt_1(:,1),mu00,V00);

end

% Quantities used in VB-M Step
data.mut = mut;
data.mutT = mutT;
data.mu0T = mu0T;
data.VtT = VtT;
data.Pt = Pt;
data.Ptt_1 = Ptt_1;
data.Vtt_1 = Vtt_1;
data.logLik = logLik;
