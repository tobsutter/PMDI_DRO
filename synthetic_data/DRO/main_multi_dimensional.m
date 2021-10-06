

%tabula rasa
close all
clear all
clc






%parameters
%****************************************************************************************************

d = 6; %dimension of E 

r = 10^(-2); %DRO radius


N_experiments = 1000;

% DESIRED ACCURACY
eps = 10^(-3);

%constraint set E
e1 = ones(d-1,1) * ((d-2)/(2*(d-1)) + 2/(3*(d-1)));
e2 = ones(d-1,1);
e3 = 0.2;
e4 = 1;
E = [e1 e2; e3 e4];



% FIND A SLATER POINT USING SOS PROGRAMMING
%C = slater_point(E);
C = 0.1; %to be found !!!!!!!!!!!!!!!!!!!!!!

%theorem parameters
D = 0.5*norm([1 1],2);

delta = 0.1;    %to be found !!!!!!!!!!!!!!!
 
%smoothing parameters
eta_1 = eps/(4*D);
eta_2 = (eps*delta^2)/(2*C^2);
eta = [eta_1,eta_2];

% operator bound
alpha = 1;    %to be found !!!!!!!!!!!


%Lipschitz constant
L = 1/eta_1 + eta_2 + (2*D + 4*D^2)^2; 



%number of iterations
N1 = 2*sqrt( (8*D*C^2)/(eps^2*delta^2) + (2*alpha^2*C^2)/(eps*delta^2) +1 )*log( (10*(eps +2*C))/eps);
N2 = 2*sqrt( (8*D*C^2)/(eps^2*delta^2) + (2*alpha^2*C^2)/(eps*delta^2) +1 )*log(C/(eps*delta*(2-sqrt(3)))*sqrt( 4*(4*D/eps + alpha^2 +eps*delta^2/(2*C^2))*(C+eps/2)));                          

N_iter = ceil(max(N1,N2))

%loss function

loss = @(beta,x,y) log( 1+exp(-y* (beta*x')) );



%different sampling numbers
N_sim = round(linspace(30,100,20));



reliability = zeros(1,length(N_sim));
Risk_sim = zeros(length(N_sim),N_experiments);
J_N_sim = zeros(length(N_sim),N_experiments);

for j=1:length(N_sim)
    disp('number of training data')
    N = N_sim(j)
    
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



%% Algorithm


%start values
w=zeros(N_iter,d);
yy=zeros(N_iter,d);

for k=1:N_iter-1
    yy(k+1,:) = w(k,:) + 1/L*gradient_eval(w(k,:),eta,N,E,d,xi);
    w(k+1,:) = yy(k+1,:)+ (sqrt(L) - sqrt(eta(2)))/(sqrt(L) + sqrt(eta(2))) * (yy(k+1,:)-yy(k,:));
end
    








%format xi = [ [x_1;y_1], [x_2;y_2], ..., [x_N;y_N] ]


    
%% recompute max_ent moments - check feasibility of computed \widehat P_N^\star
z = yy(end,:);


%1. start with x-domain

x_tr_data = xi(1:d-1,:);
y_tr_data = xi(end:end,:);

gamma = zeros(1,length(x_tr_data));

    nom = 0*ones(d,1);
    denom = 0*ones(1,1);
for i=1:length(x_tr_data)
    buf = 0;
    for jj=1:d
        buf = buf -z(jj)*xi(jj,i);
    end
        nom = nom + xi(:,i)*exp(buf);
        denom = denom + exp(buf);
        gamma(i) = exp(buf);
end
    moments_max_ent = nom./denom;
    gamma = gamma./denom;               %the gamma is needed for the predictor later

    
    
    



%% Build optimal predictor (requires solving an optimization problem) 

%parameters
param1.x_data = x_tr_data;
param1.y_data = y_tr_data;
param1.gamma = gamma;
param1.r = r;

%function for fminsearch
f = @(z) predictor_function(z,param1);

%starting point
alpha_0 = 1;
beta_0 = zeros(1,length(x_tr_data(:,1)));
z0 = [alpha_0 beta_0];

%minimize
[z,fval] = fminsearch(f,z0);


%output optimal values
alpha = z(1);
beta = z(2:end);
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

reliability = 1/N_experiments*reliability;


%plots
save('simulation_data_ME_N_10')


% for the out-of-sample risk see separate file called "plot_file.m"

%Reliability plot
figure
plot(N_sim,reliability)
title('reliability')
xlabel('radius')




    
