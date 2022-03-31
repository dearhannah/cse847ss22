function [acc] = logistic(X, y, weights)
%LOGISTIC Summary of this function goes here
%   Detailed explanation goes here
predy = X * weights >= 0;
y = y > 0;
acc = sum(y == predy)/size(y, 1);
end

