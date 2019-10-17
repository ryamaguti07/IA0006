function [J, grad] = computeCost(theta, X, y, lambda)
  m = length(y);
    h = sigmoid(X * theta);
    J = (y' * log(h) + (1 - y)' * log(1 - h)) / -m;
    grad = X' * (h - y) / m;
    th = theta; th(1) = 0;
    J = J + th' * th * lambda / m / 2;
    grad = grad + th * lambda / m;

end
