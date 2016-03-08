function sloss = sLoss(pred, outcome)

% Sum of Loss at time T for prediction
    
    % calculate Loss(prediction, outcome)
    [n, T] = size(pred);
    loss = (pred-repmat(outcome,n,1)).^2;
    % then calculate sum of Loss at time T
    sloss = loss(:,1);
    for t = 2:T
        sloss(:,t) = sloss(:,t-1)+loss(:,t);        
    end

end