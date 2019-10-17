function g = sigmoid(el)
    g = 1 ./ (1 + exp(-el));
end