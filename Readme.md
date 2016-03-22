#  Aggregating Algorithm
Codes below at AA/main.m

##  1 Import tennis1.txt dataset

    tennis1 = importfile('tennis1.txt');

##  2 Create the outcome and prediction space

    outcome = tennis1(:,2)';
    expertPrediction = tennis1(:,[4 6 8 10])';
    
##  3 Initial weight and other parameter

    [N, T] = size(expertPrediction);
    initialWeight = ones(1,N);
    eta = 2;
    
##  4 Implement Aggregating Algorithm

    pred = AA_Brier(expertPrediction, outcome, eta, initialWeight);
    
##  5 Calculate the sum of Loss 

    % for Learner
    sLL = sLoss(pred, outcome);
    sLL(length(sLL))
    ans = 1.9723e+03
    % for Expert
    sLEn = sLoss(expertPrediction, outcome);
    % for average of Expert
    sLEnAve = sLoss(mean(expertPrediction), outcome);
    
##  6 Plot the sum of Loss

    % plot LossL and LossEn
    p1 = plot(1:T, [sLL; sLEn]);
    hold on;
    title('Loss of Learner and Experts');
    legend(p1, 'Learner','Expert1','Expert2','Expert3','Expert4');
    
![image](https://raw.githubusercontent.com/vincent101/AggregatingAlgorithm/master/Result/1.jpg)
    
    % plot LossEn-LossL
    p2 = plot(1:T, [sLEn-repmat(sLL,N,1)]);
    hold on;
    title('Loss of Learner minus Loss of Experts');
    legend(p2, 'Expert1','Expert2','Expert3','Expert4');
    
![image](https://raw.githubusercontent.com/vincent101/AggregatingAlgorithm/master/Result/2.jpg)
Note: regret term  = -log(N)/eta = 0.7 in this case     
Ri(t) = LossL(t)-LossE(t) <= log(N)/eta     
    
    % plot LossEn-LossEnAve
    p3 = plot(1:T, [sLEn-repmat(sLEnAve,N,1)]);
    hold on;
    title('Loss of Expter minus Loss of average of Experts');
    legend(p3, 'Expert1','Expert2','Expert3','Expert4');
    
![image](https://raw.githubusercontent.com/vincent101/AggregatingAlgorithm/master/Result/3.jpg)
Note: regret term  = -log(N)/eta = -0.7 in this case    
Ri(t) = LossL(t)-LossE(t) <= log(N)/eta         
    
