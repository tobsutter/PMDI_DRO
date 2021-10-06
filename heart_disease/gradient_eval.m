function val = gradient_eval(z,eta,N,E,d,xi)


%inputs: eta,z, N, E, d, xi


%format xi = [ [x_1;y_1], [x_2;y_2], ..., [x_N;y_N] ]


%give back G_eta(z)

% buf_nom = 0;
% buf_denom = 0;
% 
% for j=1:N
%     buf = 0;
%     for i=1:d
%       buf = buf - z(i)*xi(i,j);
%     end
%     buf_nom = buf_nom + xi(:,j)*exp(buf);
%     buf_denom = buf_denom + exp(buf);
% end
% 




%need a stable implementation
%====================================================
isneg = zeros(length(z),1); %index for negative buuf

for j=1:N
    be(j) = 0;
    for i=1:d
      be(j) = be(j) - z(i)*xi(i,j);
    end
    al(:,j) = xi(:,j);
end
max_be = max(be);
for j=1:N
    ga(j) = be(j) - max_be;
end


%compute log of fraction
buuf = zeros(length(al(:,1)),1);
buuf2 = 0;
for j=1:N
    buuf = buuf + al(:,j) * exp(ga(j));
    buuf2 = buuf2 + exp(ga(j));
end


for i=1:d
if buuf(i)<0
    isneg(i) = 1;
    Log_nom(i) = max_be + log(-buuf(i));
else
    Log_nom(i) = max_be + log(buuf(i));
end
end

Log_denom = max_be + log(buuf2);






for i=1:d
if isneg(i)==1
    term(i) = -exp(Log_nom(i) - Log_denom);
else
    term(i) = exp(Log_nom(i) - Log_denom);
end
end
  


G_eta = - proj_E(z./eta(1),E) - eta(2)*z + (term);





%% Sanity check

% disp('stable comp')
% term'
    
% disp('sanity check - error stable')
% norm((buf_nom./buf_denom)-term')


val = G_eta;








