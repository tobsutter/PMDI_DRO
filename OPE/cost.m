function res = cost(x,a,param)

t = param.t;
v = param.v;
p = param.p;
h = param.h;

% buf = 0;
% for i=0:x+a
%     buf = buf + i*(1-t)^i*t;
% end
% 
% res = -v*buf - v*(x+a)*(1-t).^(x+a+1) + p*a + h*(x+a);
% 
% 


%Geometric distribution

res = v*(1-t)/t * ((1-t)^(x+a)-1)+ p*a + h*(x+a);


end

