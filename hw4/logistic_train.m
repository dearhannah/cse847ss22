function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%
% OUTPUT:
%    weights = (d+1) * 1 vector of weights where the weights correspond to
%              the columns of "data"
%              iterations to execute (useful when debugging in case your
%              code is not converging correctly!)
%              (if unspecified can be set to 1000)
%
[n, d_1] = size(data);
weights = zeros(d_1, 1);
acc = zeros(maxiter,1);
for iter = 1:maxiter
    [grad] = logisitc_loss_grad(data, labels, weights);
    weights = weights - epsilon .* grad;
    acc(iter) = logistic(data, labels, weights);
    % fprintf('iter %d, loss: %g \n', iter, loss);
end
plot(1:maxiter, acc);
end


function [grad] = logisitc_loss_grad(X, y, w)
[N, d_1] = size(X);
yXw = y .* (X * w);
grad = zeros(d_1, 1);
for n = 1:N
    tmp = - y(n) .* (1/(1+exp(yXw(n)))) .* X(n,:);
    grad = grad + tmp';
end
grad = grad / N;
end


function [acc] = logistic(X, Y, weights)
Y = Y > 0;
pred = X * weights >= 0;
% pred = X * weights;
acc = sum(Y == pred)/size(Y, 1);
end


