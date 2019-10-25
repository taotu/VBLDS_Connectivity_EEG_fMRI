function [etaMu_post,etaV_post,Sigma_post,a_post,b_post,etaVariance,ELBO] = state_parameter_update(data,mutT,VtT, Pt,Ptt_1,a0,b0,c_post,d_post)

sDim = size(mutT,1);    % dimension of latent variables
T = size(data.EEG,2);   % number of observations
u = data.u;     % external input
m = data.m;     % modulatory input
K = size(m,1);  % number of modulatory input


Sigma_post = zeros((K+1)*sDim+1,(K+1)*sDim+1,sDim);
etaV_post = zeros((K+1)*sDim+1,(K+1)*sDim+1,sDim);
etaMu_post = zeros((K+1)*sDim+1,sDim);
etaVariance = zeros((K+1)*sDim+1,sDim);
a_post = zeros(sDim,1);
b_post = zeros(sDim,1);

ELBO = 0;

for r = 1:sDim

    alphaMu_post = diag(c_post(:,r)./d_post(:,r));  % E(\lambda_\alpha)
    B1 = zeros((K+1)*sDim,(K+1)*sDim);
    B2 = zeros((K+1)*sDim,1);
    B3 = zeros(1,(K+1)*sDim);
    B4 = 0;
    
    A1 = zeros((K+1)*sDim,1);
    A2 = 0;
    CC = 0;
    for t = 2:T
        % Rearrange the linear equation into a compact form s(t)=eta'*Z(t)+w(t) 
        Ft = eye(sDim);
        for k = 1:K
            Ft = [Ft; m(k,t)*eye(sDim)];
        end
        % Building blocks for the Gaussian covariance
        B11 = Ft*Pt(:,:,t-1)*Ft';
        B22 = Ft*mutT(:,t-1)*u(r,t);
        B33 = u(r,t)*mutT(:,t-1)'*Ft';
        B44 = u(r,t)^2;
        
        B1 = B1+B11;
        B2 = B2+B22;
        B3 = B3+B33;
        B4 = B4+B44;
        % Building blocks for the Gaussain mean
        A11 = Ft*Ptt_1(r,:,t-1)';
        A22 = u(r,t)*mutT(r,t);
        
        A1 = A1+A11;
        A2 = A2+A22;
        
        CC = CC +log(det(VtT(:,:,t)));
    end
    
        % Update for the Gaussian posterior covariance of eta
        BB = [B1,B2;B3,B4];

        invSigma_post = BB+alphaMu_post;
        
        Sigma_post(:,:,r) = pinv(invSigma_post); % takes care zero u(t)
        
        % Update for the Gaussian posterior mean of eta
        AA = [A1;A2];
        etaMu_post(:,r) = Sigma_post(:,:,r)*AA;
        
        
        %Updata for the Gamma posteriror of beta
        C1 = sum(Pt(:,:,2:end),3);
%         a_post(r) = a0+(T-1+size(etaMu_post,1))/2;
        a_post(r) = a0+(T-1)/2;
        b_post(r) = b0+0.5*(C1(r,r)-etaMu_post(:,r)'*(invSigma_post)*etaMu_post(:,r));
        %Updata for the Gaussian posterior covariance of Sigma/beta_r
        etaV_post(:,:,r) = b_post(r)/a_post(r)*Sigma_post(:,:,r);
        etaVariance(:,r) = diag(etaV_post(:,:,r)); % variance of each parameter
        
        % Compute ELBO
        ELBO = ELBO-(T-1)/2*log(2*pi)-0.5*a_post(r)/b_post(r)*(sum(Pt(r,r,2:T)-2*etaMu_post(:,r)'*AA+...
                trace(etaMu_post(:,r)*etaMu_post(:,r)'+b_post(r)/a_post(r)*etaV_post(:,:,r)*BB)))+...
                0.5*log(det(etaV_post(:,:,r)))+((K+1)*sDim+1)/2-gammaln(a0)+a0*log(b0)-b0*a_post(r)/b_post(r)+...
                gammaln(a_post(r))-a_post(r)*log(b_post(r))+a_post(r);
end
        % Compute ELBO: the entropy of latent variables
        ELBO = ELBO+0.5*sDim*T*log(2*pi)+0.5*CC/sDim+(T-1)*sDim/2;
end
