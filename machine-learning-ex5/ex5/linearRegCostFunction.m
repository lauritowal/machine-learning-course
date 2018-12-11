function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X * theta;
error = h - y;
error_sqr = error.^2;

J = (1/(2*m)) * sum(error_sqr);

% you should not regularize the theta0 term. that why we exclude by starting with second element
thetaWithoutFirstElement = theta;
thetaWithoutFirstElement(1) = 0;
costRegularization = (lambda / (2*m)) * sum(thetaWithoutFirstElement.^2);

J = J + costRegularization;

gradRegularization = (lambda / m) * thetaWithoutFirstElement;
% grad = (1 / m) * (error' * X) + gradRegularization; % first value should not have regularization
grad = (1 / m) * (X' * error) + gradRegularization;

% =========================================================================
grad = grad(:);

end
