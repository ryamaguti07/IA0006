function [J, grad] = computeCost(theta, X, y)
    grad = zeros(size(theta));
    m = length(y);
    h = sigmoid(X * theta);
    J = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) );
    for i = 1 : size(theta, 1)
        grad(i) = (1 / m) * sum( (h - y) .* X(:, i) );
    end

end