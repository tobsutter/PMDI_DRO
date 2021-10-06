

%tabula rasa
close all
clear all
clc






%parameters
%****************************************************************************************************

d = 6; %dimension of E 

%samples

%loss function
loss = @(beta,x,y) log( 1+exp(-y* (beta*x')) );


N_experiments = 1000;
%different sample sizes
N_sim = round(linspace(30,1000,200));

reliability = zeros(length(N_sim),1);
Risk_sim = zeros(length(N_sim),N_experiments);
J_N_sim = zeros(length(N_sim),N_experiments);


for j=1:length(N_sim)
    N= N_sim(j);

%loop over all experiments
for i_ex = 1:N_experiments


%% Generate training data

for ii=1:N
    x(:,ii) = unifrnd(0,1,d-1,1);
    buf = 0;
    for j1=1:d-1
        buf = buf + x(j1,ii);
    end
    if 1/(d-1)*buf >=1/2
        y(1,ii) = 1;
    else y(1,ii) = -1;
    end
end
        
xi = [x;y];
%format xi = [ [x_1;y_1], [x_2;y_2], ..., [x_N;y_N] ]


%1. start with x-domain
x_tr_data = xi(1:d-1,:);
y_tr_data = xi(end:end,:);



%% Build optimal predictor (requires solving an optimization problem) 

%parameters
param1.x_data = x_tr_data;
param1.y_data = y_tr_data;

%function for fminsearch
f = @(z) predictor_function_IWERM(z,param1);

%starting point
beta_0 = 0*ones(1,length(x_tr_data(:,1)));
z0 = beta_0;

%minimize
[z,fval] = fminsearch(f,z0);


%output optimal values
beta = z;
J_N = fval;  %certificate





%%

%test distribution

%evaluate risk
%====================================
N_MC = 10000; %Monte Carlo samples
x_samples = unifrnd(0,1,length(beta),N_MC);
buf = 0;

for ii=1:N_MC
   x_s = x_samples(:,ii);
   cond = 2/length(x_s)*sum(x_s);
   if cond > 1/2
       buf = buf + int_risk_1(beta,x_s,1);
   else
       buf = buf + int_risk_1(beta,x_s,-1);
   end
end
   
risk = 1/N_MC * buf;


%save simulation output
J_N_sim(j,i_ex) = J_N;
Risk_sim(j,i_ex) = risk;

%reliability


if Risk_sim(j,i_ex)<= J_N_sim(j,i_ex)
    reliability(j) = reliability(j)+1;
end

end
end

%Prepare plot
reliability = 1/N_experiments*reliability;

DataMatrix = Risk_sim';



save('simulation_data_IWERM_varions_N_dim5')



     

%Reliability plot
figure
plot(N_sim,reliability)
title('reliability')
xlabel('radius')
    
