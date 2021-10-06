function val = predictor_function(z,param)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

x_data = param.x_data;
y_data = param.y_data;
gamma = param.gamma;
r = param.r;

x_data_tot = param.x_data_tot;

alpha = z(1);
beta = z(2:end)';

%beta = param.beta;


beta = beta';


%find L_max
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_data = length(x_data_tot(1,:));
buf = -10^6;
for i=1:N_data
    x_s = x_data_tot(:,i);
    L_max1 = log(1+exp(1*beta*x_s));
    L_max2 = log(1+exp(-1*beta*x_s));
    L_max = max(L_max1,L_max2);
    if L_max > buf
        buf = L_max;
    end
end
L_max_final = buf;


% %find L_max
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N = 10000; %Monte carlo samples
% x_samples = unifrnd(0,1,length(beta),N);
% buf = -10^6;
% for i=1:N
%     x_s = x_samples(:,i);
%     L_max1 = max(log(1+exp(1*beta*x_s)));
%     L_max2 = max(log(1+exp(-1*beta*x_s)));
%     L_max = max(L_max1,L_max2);
%     if L_max > buf
%         buf = L_max;
%     end
% end
% 
% L_max_final = buf;


%evaluate function

buf = 1;


for i=1:length(x_data)
    buf = buf*(alpha - log(1 + exp(-y_data(i)*(beta*x_data(:,i))))).^(gamma(i));
end

%check feasibility of alpha
if alpha >= L_max_final
    val = alpha - exp(-r)*buf;
else
    val = 10^10;
end




end

