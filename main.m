% import data set
tennis1 = importfile('tennis1.txt');
% create outcome space, 1*T
outcome = tennis1(:,2)';
% create prediction space of expert, N*T 
expertPrediction = tennis1(:,[4 6 8 10])';
% initial weight
[N, T] = size(expertPrediction);
initialWeight = ones(1,N);
% implement algorithm
eta = 2;
pred = AA_Brier(expertPrediction, outcome, eta, initialWeight);

% calculate the sum of Loss for Learner, Expert and ExpertAverage
sLL = sLoss(pred, outcome);
sLEn = sLoss(expertPrediction, outcome);
sLEnAve = sLoss(mean(expertPrediction), outcome);

% plot LossL and LossEn
p1 = plot(1:T, [sLL; sLEn]);
hold on;
title('Loss of Learner and Experts');
legend(p1, 'Learner','Expert1','Expert2','Expert3','Expert4');
% plot LossEn-LossL
p2 = plot(1:T, [sLEn-repmat(sLL,N,1)]);
hold on;
title('Loss of Learner minus Loss of Experts');
legend(p2, 'Expert1','Expert2','Expert3','Expert4');
% plot LossEn-LossEnAve
p3 = plot(1:T, [sLEn-repmat(sLEnAve,N,1)]);
hold on;
title('Loss of Expter minus Loss of average of Experts');
legend(p3, 'Expert1','Expert2','Expert3','Expert4');