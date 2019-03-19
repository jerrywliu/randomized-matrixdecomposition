% Test variables
n = 200;
iters = 5;
sigsings = 10;
omega = normrnd(0,1,[n,n]);
[U,S,V] = svd(omega);
singularvalues = diag(S);
for i = sigsings+1:n
    singularvalues(i) = singularvalues(i)/1e5;
end
omega = U*diag(singularvalues)*V.';
semilogy(singularvalues);

%{
% Test of iteratedranger
singularvalues = singularvalues(1:n);
E = zeros(n-1);
for i = 1:size(E)
    Q = iteratedranger(omega,i,iters);
    E(i) = norm((eye(200)-Q*Q.')*omega);
end
hold on;
plot(E);
hold off;
%}

%{
% Test of randomized SVD
Q = ranger(omega,sigsings);
[U,S,V] = directsvd(omega,Q);
omegaest = U*S*V.';
norm(omega)
norm(omegaest)
norm(omega-omegaest)
hold on;
plot(diag(S));
hold off;
%}


% Test of sketchSVD
[~,Y,psi,W] = sketch(omega,2*sigsings,2*sigsings);
[U,S,V] = sketchsvd(Y,psi,W);
omegaest = U*S*V.';
norm(omega,'fro')
norm(omegaest,'fro')
norm(omega-omegaest,'fro')
hold on;
plot(diag(S),'*-');
hold off;


% Given m x n matrix A, integer k
% Computes m x k orthonormal matrix Q whose range approximates range of A
function Q = ranger(A,k)
    [~,n] = size(A);
    omega = normrnd(0,1,[n,k]);
    Y = A*omega;
    [Q,~] = qr(Y,0);
end

% Given m x n matrix A, tolerance eps, integer r
% Computes orthonormal matrix Q such that norm((eye(n)-Q*Q.')*A) <= eps
% with probability at least 1 - min(m,n)*10^-r
function Q = adaptiveranger(A,eps,r)
    [~,n] = size(A);
    Y = zeros([n,r]);
    maxnorm = 0;
    for i = 1:r
        omega = normrnd(0,1,[n,1]);
        Y(:,i) = A*omega;
        if norm(Y(:,i)) > maxnorm
            maxnorm = norm(Y(:,i));
        end
    end
    Q = zeros(n,0);
    j = 0;
    while maxnorm > eps/(10*sqrt(2/pi))
        j = j + 1;
        Y(:,j) = (eye(n)-Q*Q.')*Y(:,j);
        Q = [Q, (Y(:,j)/norm(Y(:,j)))];
        omega = normrnd(0,1,[n,1]);
        Y = [Y, (eye(n)-Q*Q.')*A*omega];
        maxnorm = norm(Y(:,j+r));
        for i = 1:r-1
            Y(:,j+i) = Y(:,j+i)-Q(:,j)*(Q(:,j).'*Y(:,j+i));
            if norm(Y(:,j+i)) > maxnorm
                maxnorm = norm(Y(:,j+i));
            end
        end
    end
end

% Given m x n matrix A, integer k, integer q
% Computes m x p orthonormal matrix Q whose range approximates range of A
% Applicable for A whose singular values decay slowly
function Q = iteratedranger(A,k,q)
    [~,n] = size(A);
    omega = normrnd(0,1,[n,k]);
    Y = A*omega;
    [Q,~] = qr(Y,0);
    for i = 1:q
        Y = A.'*Q;
        [Q,~] = qr(Y,0);
        Y = A*Q;
        [Q,~] = qr(Y,0);
    end
end

% Given m x n matrix A, integer k, integer p
% Computes m x k matrix Y whose range approximates range of A
% and p x n matrix W whose corange approximates corange of A
function [omega,Y,psi,W] = sketch(A,k,p)
    [m,n] = size(A);
    omega = randn(n,k);
    psi = randn(p,m);
    omega = orth(omega);
    psi = orth(psi.').';
    Y = A*omega;
    W = psi*A;
end

% Given m x k orthonormal matrix Q whose range approximates range of A
% and m x n matrix A
% Computes an approximate SVD A ~= U*S*V.'
% U and V are orthonormal, S is a nonnegative diagonal matrix
function [U,S,V] = directsvd(A,Q)
    B = Q.'*A;
    [Ub,S,V] = svd(B);
    U = Q*Ub;
end

% Given m x k matrix Y whose range approximates range of A
% p x n matrix W whose corange approximates corange of A
% and p x m random normal matrix psi such that psi*A = W
% Computes an approximate SVD A ~= U*S*V.'
% U and V are orthonormal, S is a nonnegative diagonal matrix
function [U,S,V] = sketchsvd(Y,psi,W)
    [Q,~] = qr(Y,0);
    [Qq,Rq] = qr(psi*Q);
    B = pinv(Rq)*Qq'*W;
    [Ub,S,V] = svd(B);
    U = Q*Ub;
end