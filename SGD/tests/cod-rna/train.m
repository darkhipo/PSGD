function train(nIters, eta)

load fisheriris;

X = meas(1:100,:);
y = zeros(100,1);
y(51:100) = ones(50,1);


%nIters = 100;
%eta = 0.0001;
%[loglik, w] = lrbgd(X, y, eta, nIters);
%w
[n,d] = size(X);
XX = zeros(100*nIters,d);
yy = zeros(100*nIters,1);
for i = 1 : nIters
    idx = randperm(100);
    startIdx = (i-1)*100 + 1;
    endIdx = i*100;
    XX(startIdx:endIdx,:) = X(idx,:);
    yy(startIdx:endIdx) = y(idx);
end

dlmwrite('examples',XX, 'delimiter','\t','precision',6);
dlmwrite('labels',yy, 'delimiter', '\t');
[loglik, w] = lrsgd(XX, yy, eta);
w
sum(loglik)/size(loglik,1)
