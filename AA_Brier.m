function pred = AA_Brier(expertPrediction, outcome, eta, initialWeight)

% Implement Aggregating Algorithm, with expert prediction space, 
% outcome space, parameter eta and initial distribution

    [N, T] = size(expertPrediction);
    % normalise the initial weight
    weight = initialWeight/sum(initialWeight);
    
    for t = 1:T
        % read expert prediction in time t
        ePred = expertPrediction(:,t);
        % normalise the weight
        p = weight;
        % solve the system
        g0 = -1/eta * log(p*exp(-eta*ePred.^2));
        g1 = -1/eta * log(p*exp(-eta*(1-ePred).^2));
        pred(t) = (1-g1+g0)/2; %#ok<AGROW>
        % read the outcome
        w = outcome(:,t);
        % update the weight
        weight = weight.*exp(-eta*(ePred-w).^2)'; 
        % normalise weight, avoiding weight goes down to zero
        weight = weight/sum(weight);            
    end
    
end