%% X : n * d, y:n*1, w : d*1
function [loglik, w] = lrsgd(X, y, eta)

[n,d] = size(X);
x = zeros(d,1);
w = zeros(d,1);
grad = zeros(d,1);
est = zeros(n,1);
loglik = zeros(n, 1);
sumLoglik = zeros(n,1);
wnorm = zeros(n,1);
gradNorm = zeros(n,1);
weight1 = zeros(n,1);
weight2 = zeros(n,1);
weight3 = zeros(n,1);
weight4 = zeros(n,1);
for i = 1 : n
    x = (X(i,:))'; % example i, d*1
    est = sigmoid(x'*w);
    grad = x * (est - y(i)); % d*1
    wnew = w - eta* grad;
    wnorm(i) = norm(w-wnew,2);
    gradNorm(i) = norm(grad,2);
    w = wnew;
    weight1(i) = w(1);
    loglik(i) = y(i)*log(est) + (1 - y(i))*log(1 - est);
    if (i > 1)
        sumLoglik(i) = sumLoglik(i-1) + loglik(i);
    else
        sumLogLik(i) = loglik(i);
    end
end

figure;
plot([1:n], weight1);
title('weight 1');


figure;
plot([1:n], wnorm);
title('weight diff norm');

figure;
semilogx([1000:n], gradNorm(1000:n));
title('grad norm');

figure;
plot([1:n], loglik);
title('loglik');


function a = sigmoid(z)
a = 1.0 ./ (1.0 + exp(-z));




