function [data] = VBLDS_initialize(data)

u = data.u;     % external stimuli
m = data.m;     % modulatory input: each row represents one input
L = data.L;     % Lead-field matrix

G = data.G;     % source membership matrix
C = L*G;        % Emission matrix
sDim = size(G,2);       % number of ROIs (latent variable)
K = size(m,1);      % number of modulatory inputs
yEEG = data.EEG;        % EEG observations
T = size(yEEG ,2);      % number of observations
yDim = size(yEEG,1);        % number of EEG channels

% EEG sources from the inverse solution
s = (C'*C+5e-8*eye(size(C,2)))\C'*yEEG;

% Solve linear regression on observation equation to estimate latent sources 
Qy0 = data.Qy0;
Qx0 = data.Qx0;
Re0 = Qy0+L*Qx0*L';

% Solve the initial solutions for model parameters in the state equation
Y = s';
               
d0 = zeros(sDim,1);
Q0 = zeros(sDim,sDim);
A0 = zeros(sDim);
B0 = zeros(sDim,sDim,K);

for r = 1:sDim
    
     if sum(u(r,:)) ~= 0
        % Get regressors
        popt = 1;
        Xo = getRegressors(Y,popt);   % Regressors from data time shift by x
        ur = u(r,:)';
        X = Xo;      
        for k = 1:K
            u2 = m(k,:)';
            U2 = u2(popt+1:end) * ones(1,sDim);
            X1 = Xo.*U2;
            X = [X X1];
        end
        X = [X ur(popt+1:end)];    %   %Include External stimuli as a regressor

        yy = Y(2:end,r);
%         w = X\y;
        w = pinv(X)*yy;
        A0(r,1:sDim) = w(1:sDim);
        ix = sDim+1;
        for k = 1:K
            B0(r,1:sDim,k) = w(ix:ix+sDim-1);
            ix = ix + sDim;
        end
        d0(r) = w(end);
        Q0(r,r) = ((yy-X*w)'*(yy-X*w))/(T-length(w));     
     else
      % Get regressors
        popt = 1;
        Xo = getRegressors(Y,popt);   % Regressors from data time shift by x
        X = Xo;       
        for k = 1:K
            u2 = m(k,:)';
            U2 = u2(popt+1:end) * ones(1,sDim);
            X1 = Xo.*U2;
            X = [X X1];
        end
        yy = Y(2:end,r);
        w = pinv(X)*yy;
        A0(r,1:sDim) = w(1:sDim);
        ix = sDim+1;
        for k = 1:K
            B0(r,1:sDim,k) = w(ix:ix+sDim-1);
            ix = ix + sDim;
        end
        d0(r) = 0;
        Q0(r,r) = ((yy-X*w)'*(yy-X*w))/(T-length(w));
     end
end

D0 = diag(d0);

% Initialization of model parameters
data.init.A0 = A0;
data.init.B0 = B0;
data.init.D0 = D0;
data.init.Re0 = Re0;
data.init.Q0 = Q0;
data.init.Qx0 = Qx0;
data.init.Qy0 = Qy0;


% Initialization of hyper parameters
data.init.a0 = 10^(-4);     % a0<b0 to induce informative prior on the state noise precision
data.init.b0 = 10^(-3);
data.init.c0 = 10^(-2);
data.init.d0 = 10^(-4);     % c0>do to impose more sparsity
data.init.v0 = yDim+1;


data.maxIter = 60;      % maximum number of iterations
data.tol = 10.^-4;      % tolerance for converence
data.flag = 0;      % monitor if R has negative elements
data.posteriorEStepFlag = 0;        % set to 0 to use initial guess
data.posteriorMStepFlag = 0;        % set to 0 to use initial guess

% Initial state and state covariance for Kalman filter
data.init.s0 = zeros(sDim,1);
data.init.Q0KL = 1*eye(sDim);

end

function X = getRegressors(Y,p)

[N,M] = size(Y);
X = [];
L = N-p;
for lag = p:-1:1
    for m = 1:M
        x = Y(lag:lag+L-1,m);
        X = [X x];
    end
end

end