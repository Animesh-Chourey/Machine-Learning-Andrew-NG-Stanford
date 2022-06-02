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


g=sigmoid(X*theta);
mul1=y.*log(g);
mul2=(1-y).*(log(1-g));
div=-(mul1+mul2)/m;
part1=sum(sum(div)); %Calculated the first part of the eqn

s=length(theta);
part2=(lambda/(2*m))*sum(theta(2:s,:).^2); %Calculated the second part of the eqn

J=part1+part2;

tempTheta = theta;
tempTheta(1) = 0; %Since we don't change the theta(1) value setting this to 0 will help in not changing the vlue of theta(1) position

diff=g-y;
mul3=X'*diff;
grad=mul3*(1/m)+(lambda/m)*tempTheta;


% =============================================================

end
