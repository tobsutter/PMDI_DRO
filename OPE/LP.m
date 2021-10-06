function [A,b] = LP(param)

x_grid = param.x_grid;
a_grid = param.a_grid;
n = param.n;
m = param.m;

A1 = zeros(n,m*n);
for i=1:n
    A1(i,(i-1)*m+1:i*m)=ones(1,m);
end

A2 = zeros(n,n*m);
for y=1:n
    for i=1:n
        for j=1:m
            A2(y,(i-1)*m+j) = Q(x_grid(y),x_grid(i),a_grid(j),param);
        end
    end
end

A = A1 - A2;
b = zeros(n,1);