function [c_post,d_post,ELBO] = state_hyperparameter_update(c0,d0,etaMu_post,Sigma_post,a_post,b_post)

J = size(Sigma_post,1);     % number of hyperparameters (K+1)S+1
sDim = size(a_post,1);      % number of EEG sources

c_post = zeros(J,sDim);
d_post = zeros(J,sDim);

ELBO = 0;

for r = 1:sDim
    for j = 1:J        
        c_post(j,r) = c0+1/2;
        d_post(j,r) = d0+0.5*(a_post(r)/b_post(r)*(etaMu_post(j,r)^2)+Sigma_post(j,j,r));
    end
    ELBO = ELBO -J*gammaln(c0)+J*c0*log(d0)+J*gammaln(c0+1/2)-(c0+1/2)*sum(log(d_post(j,r)));
end

end