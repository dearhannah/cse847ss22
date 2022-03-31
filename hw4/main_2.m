function main_2()
% Specify the options (use without modification).
opts.rFlag = 1;  % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4;  % termination options.
opts.maxIter = 5000; % maximum iterations.
load('ad/ad_data.mat');

% [w, c] = LogisticR(data, labels, par, opts);

for par = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    if par > 0
        [w, c] = LogisticR(X_train, y_train, par, opts);
    else
        w = logistic_train([X_train, ones(size(X_train, 1), 1)], y_train, 5e-2, 5000);
        c = w(size(w, 1));
        w = w(1:size(w, 1)-1);
    end
    preds = X_test * w + c;
%     y = y_test > 0;
    [~, ~, ~, auc] = perfcurve(y_test, preds, 1);
    fprintf('par: %g, auc: %g, number of features: %d\n', par, auc, sum(abs(w)>1e-12, 1));
end

end

% par: 0, auc: 0.702392, number of features: 116
% par: 0.01, auc: 0.629665, number of features: 105
% par: 0.1, auc: 0.698565, number of features: 14
% par: 0.2, auc: 0.679426, number of features: 5
% par: 0.3, auc: 0.644976, number of features: 3
% par: 0.4, auc: 0.622967, number of features: 2
% par: 0.5, auc: 0.62201, number of features: 1
% par: 0.6, auc: 0.62201, number of features: 1
% par: 0.7, auc: 0.62201, number of features: 1
% par: 0.8, auc: 0.62201, number of features: 1
% par: 0.9, auc: 0.62201, number of features: 1
% par: 1, auc: 0.5, number of features: 0
