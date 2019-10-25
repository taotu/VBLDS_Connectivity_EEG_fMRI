function [mutT,VtT,Pt,Ptt_1,mu0T] = KL_smoothing_backward(A,B,m,D,u,mut,Vt,Vtt_1,V10,mu10,mu00,V00)

S = size(Vt,1);     % dimension of the latent state
T = size(Vt,3);% number of observations

mutT = zeros(S,T);
mutT(:,T) = mut(:,T);

K = size(m,1);

VtT = zeros(S,S,T);
VtT(:,:,T) = Vt(:,:,T);

Jt = zeros(S,S,T-1);

Pt = zeros(S,S,T);
Ptt_1 = zeros(S,S,T-1);

%%%% Backward sequential update %%%%%
for t = T-1:-1:1
    Bmt = zeros(size(A));
    for k = 1:K
        Bmt = Bmt+m(k,t+1).*B(:,:,k);
    end
    G = A+Bmt;
    % Kalman gain 
      Jt(:,:,t) = Vt(:,:,t)*G'/(Vtt_1(:,:,t+1));
    
    % Kalman smoothing
    mutT(:,t) = mut(:,t)+Jt(:,:,t)*(mutT(:,t+1)-G*mut(:,t)-D*u(:,t+1));
    VtT(:,:,t) = Vt(:,:,t)+Jt(:,:,t)*(VtT(:,:,t+1)-Vtt_1(:,:,t+1))*Jt(:,:,t)';
    
    % Expectations for VB-M step
    Pt(:,:,t) = VtT(:,:,t)+ mutT(:,t)*mutT(:,t)';
    Ptt_1(:,:,t) = Jt(:,:,t)*VtT(:,:,t+1)+ mutT(:,t+1)*mutT(:,t)';
    
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pt(:,:,T) =  VtT(:,:,T)+ mutT(:,T)*mutT(:,T)';
% The kalman smoother for mu0 (used to estimate Q0 in KL initialization)
mu0T = mu00+(V00*G')/V10*mu10;