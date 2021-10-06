function res = gradient_eval(z,A,b,eta,q,param)

n = param.n;
m = param.m;

eta2 = eta(2);

z = z';

y = -A'*z;
buf = 0;

for i=1:n*m
    v(i) = exp(y(i))*q(i);
end

s = sum(v);
mu = (v./s)';

res = -b + A*mu - eta(2)*z;

res = res';



end

