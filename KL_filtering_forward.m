function [mut,mutt_1,Vt,Vtt_1,logLik] = KL_filtering_forward(x,u,A,B,m,s0,D,Q,Q0,L,R)

S = size(s0,1);     % dimension of the latent state
C = size(x,1);      % dimension of the observation 
T = size(x,2);      % number of observations
K = size(m,1);      % number of modulatory input
mut = zeros(S,T);
mut = [s0 mut];
mutt_1 = zeros(S,T);

Vt = zeros(S,S,T);
Vt = cat(3,Q0,Vt);
Vtt_1 = zeros(S,S,T);
Kt = zeros(S,C,T);
logLik = 0;
%%%% Forward sequential update %%%%%
for t = 1:T
    Bmt = zeros(size(A));
    for k = 1:K
        Bmt = Bmt+m(k,t).*B(:,:,k);
    end
    G = A+Bmt;
    % Prediction for state vector and covariance
    Vtt_1(:,:,t) = G*Vt(:,:,t)*G'+Q;
    mutt_1(:,t) = G*mut(:,t)+D*u(:,t);
    
    % Kalman gain
    Kt(:,:,t) = Vtt_1(:,:,t)*L'/(R+L*Vtt_1(:,:,t)*L');
    
    % Correction based on observation
    Vt(:,:,t+1) = Vtt_1(:,:,t)-Kt(:,:,t)*L*Vtt_1(:,:,t);    
    mut(:,t+1) = mutt_1(:,t)+Kt(:,:,t)*(x(:,t)-L*mutt_1(:,t)); 

    % Compute the log likelihood function
    H = L*Vtt_1(:,:,t)*L'+R;
    sigma = 0.5*(H+H');
    logLik = logLik+log_gaussian_likelihood(x(:,t)',(L*mutt_1(:,t))',sigma);
end

mut = mut(:,2:end);
Vt = Vt(:,:,2:end);

end

function LL = log_gaussian_likelihood(X,mu,sigma)
    
 [R, err] = cholcov(sigma, 0);

if err
    error('%s', 'sigma is not both symmetric and positive definite');
end

    X0 = bsxfun(@minus, X, mu) / R;
    d = min(size(X));
    slogdet = sum(log(diag(R)));
    LL = -0.5 * sum(X0 .^ 2, 2) - slogdet - 0.5 * d * log(2 * pi);
end
