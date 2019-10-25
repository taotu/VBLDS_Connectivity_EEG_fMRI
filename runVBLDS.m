function [data] = runVBLDS(data)

% Model initialization 
data = VBLDS_initialize(data);

% Iterate between VB-EStep and VB-MStep until convergence

    LBPrevious = -inf;
    convergeFlag = 0;
    iter = 1;
    logLh = [];

    while ~convergeFlag && (iter <= data.maxIter)
    if mod(iter,10)==0
       disp(['iteration step #' num2str(iter)]);
    end
    % ############################ E-Step ############################
        data_old = data;
        data = VB_EStep(data);
    % ############################ M-Step ############################
        data = VB_MStep(data);
        LB = data.ELBO;  % ELBO in this iteration
        % Check for convergence
        if iter >= 0
            [convergeFlag, decreased] = em_converged(LB, LBPrevious, data.tol,1);
            
           if decreased || data.flag 
                data = data_old;
                convergeFlag = 1;
                iter = iter + 1;
                logLh = [logLh LB];

            else
                LBPrevious = LB;
                iter = iter + 1;
                logLh = [logLh LB];
            end
            
        else
            logLh = [logLh LB];
            iter = iter + 1;
        end


    end


data.logLh = logLh;

end



function [converged, decrease] = em_converged(loglik, previous_loglik, threshold, check_increased)
% EM_CONVERGED Has EM converged?
% [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
%
% We have converged if the slope of the log-likelihood function falls below 'threshold',
% i.e., |f(t) - f(t-1)| / avg < threshold,
% where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
% 'threshold' defaults to 1e-2.
%

if nargin < 3, threshold = 1e-2; end
if nargin < 4, check_increased = 1; end

converged = 0;
decrease = 0;

if check_increased
    if loglik - previous_loglik < -1e-3 % allow for a little imprecision
        fprintf(1, '******likelihood decreased from %6.4f to %6.4f!\n', previous_loglik, loglik);
        decrease = 1;
        converged = 0;
        return;
    end
end

delta_loglik = abs(loglik - previous_loglik);
avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2;
if (delta_loglik / avg_loglik) < threshold, converged = 1; end
end