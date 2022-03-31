function main_1()
data = load('spam_email/data.txt');
labels = load('spam_email/labels.txt');
epsilon = 5e-2;
maxiter = 1000;
labels(labels==0) = -1;

if size(data, 1) == 57
    data = [data, ones(size(data, 1))];
end
acc = zeros(6,1);
N = [200, 500, 800, 1000, 1500, 2000];
for i = 1:6
    n = N(i);
    [weights] = logistic_train(data(1:n, :), labels(1:n), epsilon, maxiter);
    acc(i) = logistic(data(2001:4601, :), labels(2001:4601), weights);
    disp([num2str(n), ': ', num2str(acc(i))]);
end
plot(N,acc)
end


function [acc] = logistic(X, Y, weights)
Y = Y > 0;
pred = X * weights >= 0;
% pred = X * weights;
acc = sum(Y == pred)/size(Y, 1);
end

% 200: 0.88889
% 500: 0.90042
% 800: 0.89927
% 1000: 0.90119
% 1500: 0.90196
% 2000: 0.90388
