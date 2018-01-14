function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%theta_1 = theta(2:end);
%X_1=X(:,2:end);
%hypothesis = sigmoid(X*theta);
%h = sigmoid(X_1*theta_1);
%J = (sum(-y'*log(hypothesis)-(1-y')*log(1-hypothesis)))/m + (lambda/(2*m))*(sum(theta_1).^2) ;

%grad_0 = (X' * (hypothesis-y))/m ;
%%grad_1 = (X_1' * (hypothesis-y))/m + (lambda/m)*sum(theta_1);
%grad_1 = (sum(X_1' * (hypothesis-y)))/m + ((lambda*sum(theta_1))/m);
%grad=vertcat(grad_0,grad_1);

[J, grad] = costFunction(theta, X, y);

% this effectively ignores "theta zero" in the following calculations
theta_0 = [0; theta(2:end);];

J = J + lambda / (2 * m) * sum( theta_0 .^ 2 );
grad = grad .+ (lambda / m) * theta_0;



% =============================================================

end
