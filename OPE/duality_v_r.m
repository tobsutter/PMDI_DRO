function res = duality_v_r(alpha, mu,c, param)


n = param.n;
m = param.m;
r = param.r;

gamma_bar = max(c);

buf = 1;

for i=1:n*m
    buf = buf*(alpha - c(i))^mu(i);
end

if alpha >= gamma_bar
    res = alpha - exp(-r)*buf;
else 
    res = 10^6;
end


end

