%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OFF POLICY EVALUATION - INVENTORY MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%tabula rasa
clear all
close all
clc



%define parameters

param.n = 6; % #states  X={0,1,2, .... , n-1}
n = param.n;
param.x_grid = 0:n-1;
x_grid = param.x_grid;
param.m = 4; % #actions A={0,1,2, .... , m-1}
m = param.m;
param.a_grid = 0:m-1;
a_grid = param.a_grid;

param.t = 0.2; %parameter of the geometric distribution modelling the demand

param.v = 1;   %sales price
param.p = 0.6; %purchase price
param.h = 0.3; %holding cost


param.beta = 4; %propensitiy regularization parameter




T_vec = linspace(10,400,5);



param.r = 10^(-2);    %DRO radius

N_ex = 5; % #experiments

w_noise = random('geo',param.t,1,10^6);

%% Choose the policies (randomly)
%pick random evaluation policy
for i=1:n
    r = rand(m,1);
    r = r./sum(r);
    Pi_e(i,:) = r';
end

%random behavioural policy
for i=1:n
    r = rand(m,1);
    r = r./sum(r);
    Pi_b(i,:) = r';
end


%% Choose evaluation occupation measure
T_eval = 10^5;

%simulate occupation measure
x_last=0; %initial condition
x_traj = [];
a_traj = [];
occ = zeros(n,m);
cost_traj = [];
for tt = 1:T_eval
    a_distr = Pi_e(x_last+1,:); %select randomized policy
    a = gendist(a_distr,1,1)-1; %draw policy realization
    a_traj = [a_traj; a];
    w = w_noise(tt); %draw random demand    
    x_next = max(0,min(n-1,x_last+ a)-w);
    x_traj = [x_traj;x_next];
    cost_traj = [cost_traj; cost(x_next,a,param)];
    x_last = x_next;
    
    %compute occupation measure
    for i=1:n
        for j=1:m
            if x_next == x_grid(i) && a == a_grid(j)
                occ(i,j) = occ(i,j)+1;
            end
        end
    end
 
end
occ_e = 1/T_eval.*occ; %empirical behavioural occupation measure 

occ_e = occ_e + ones(size(occ_e)).*10^(-15); %avoid singularities

%% True (unknown) evaluation cost

c_eval = 0;

x = 0; %initial condition
a = 0;

for tt=1:T_eval
    a_distr = Pi_e(x+1,:); %select randomized policy
    a = gendist(a_distr,1,1)-1; %draw policy realization
    w = w_noise(tt); %draw random demand
    x = max(0,min(n-1,x+ a)-w);
    c_eval = c_eval + cost(x,a,param);
end

%disp('true (unknown) evaluation cost')
V_e = c_eval/T_eval;


disap_DRO = zeros(length(T_vec),N_ex);
disap_IPS = zeros(length(T_vec),N_ex);
disap_IPS_beta = zeros(length(T_vec),N_ex);

for t_s = 1:length(T_vec)
    T_sim = T_vec(t_s);
    disp('#samples:')
    T_sim

for i_ex = 1:N_ex
    disp('Experiment:')
    i_ex
    
    %generate data
    w_noise = random('geo',param.t,1,10^6);


%% Behavioural occupation and cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Step 1: find true occupation meausre


%behavioual occupupation measure and cost
x_last=0; %initial condition

cost_matrix_b = zeros(n,m);
x_traj = [];
a_traj = [];
occ = zeros(n,m);
cost_traj = [];
for tt = 1:T_sim
    a_distr = Pi_b(x_last+1,:); %select randomized policy
    a = gendist(a_distr,1,1)-1; %draw policy realization
    a_ind = find(a==a_grid);
    a_traj = [a_traj; a];
    w = w_noise(tt); %draw random demand
    x_next = max(0,min(n-1,x_last+ a)-w);
    x_ind = find(x_next == x_grid);
    x_traj = [x_traj;x_next];
    cost_traj = [cost_traj; cost(x_next,a,param)];
    x_last = x_next;
    cost_matrix_b(x_ind,a_ind)=cost_matrix_b(x_ind,a_ind)+cost(x_next,a,param);
    
    %compute occupation measure
    for i=1:n
        for j=1:m
            if x_next == x_grid(i) && a == a_grid(j)
                occ(i,j) = occ(i,j)+1;
            end
        end
    end
 
end

occ = occ + ones(size(occ)).*10^(-15); %avoid singularities

occ_b = 1/T_sim.*occ; %empirical behavioural occupation measure   
cost_matrix_b = cost_matrix_b./T_sim;


    
%% Inverse propensity weights

we = occ_e./occ_b;
%disp('Inverse propensity method')
V_IPS(t_s, i_ex) = sum(sum(cost_matrix_b.*we));

%regularized IPS
V_IPS_beta(t_s, i_ex) = sum(sum(cost_matrix_b.*min(we,ones(size(we))*param.beta)));








%% DRO method

%Generate occupation measure LP-Matrices
[A,b] = LP(param);


%compute the projection f_\Pi(\mu_b)
buf = 0;
q = zeros(n*m,1);
for i=1:n
    for j=1:m
        buf = buf + occ_e(i,j)*log(occ_e(i,j)/occ_b(i,j));
        a1((i-1)*m+j) = log(occ_e(i,j)/occ_b(i,j));
        q((i-1)*m+j) = occ_b(i,j);
    end
end
alpha = buf;

A1(1,:) = a1;
b1 = [alpha];
A2 = A;
b2 = b;
mub = occ_b;

A = [A1;A2];
b = [b1;b2];

%minmize
% min_p f(p) st. Ap = b


% Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_iter = 10^6;

% %find optimal smoothing parameters
% 
% C = log(1/(min(min(occ_b))));
% D = 0.5*norm(b);
% alpha = norm(A);
% de = 10^(-0.2);
% 
% f_iter = @(ep) N_iter - max(2*(sqrt( 8*D*C^2/(ep^2*de^2) + 2*alpha*C^2/(ep*de^2)+1 ))*log(10*(ep+2*C)/(ep)), 2*(sqrt( 8*D*C^2/(ep^2*de^2) + 2*alpha*C^2/(ep*de^2)+1 ))*log(C/(ep*de*(2-sqrt(3)))*sqrt(4*(4*D/ep + alpha^2 + ep*de^2/(2*C^2))*(C+ep/2)))); 
% 
% ep_opt = fsolve(f_iter,0.1);
% 
% ep_opt
% if real(ep_opt)<= 0
%     ep_opt = 0.01;
% end
% 
% eta1 = real(ep_opt/(4*D))
% eta2 = real(ep_opt*de^2/(2*C^2))

%suboptimal choise or eta (Tuning)
eta1 = 10^(-1);
eta2 = 10^(-2);
eta = [eta1; eta2];

L = 1/eta1 + norm(A)^2 + eta2;



w=zeros(N_iter,n+1);
yy=zeros(N_iter,n+1);

for k=1:N_iter-1
    yy(k+1,:) = w(k,:) + 1/L*gradient_eval(w(k,:),A,b,eta,q,param);
    w(k+1,:) = yy(k+1,:)+ (sqrt(L) - sqrt(eta(2)))/(sqrt(L) + sqrt(eta(2))) * (yy(k+1,:)-yy(k,:));
end


z = yy(end,:)';
%reconstruct optimal distribution
y = -A'*z;
buf = 0;
for i=1:n*m
    v(i) = exp(y(i))*q(i);
end
s = sum(v);
p_star = (v./s)';



% DRO optimization (from data to decision paper - duality formula)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%generate vectors
c_vec = zeros(n*m,1);
p_e = zeros(n*m,1);
for i=1:n
    for j=1:m
        x = x_grid(i);
        a = a_grid(j);
        c_vec((i-1)*m+j) = cost(x,a,param);
        p_e((i-1)*m+j) = occ_e(i,j);
    end
end




%function for fminsearch
fun = @(alpha) duality_v_r(alpha, p_star ,c_vec, param);

%starting point
alpha_0 = max(c_vec)+10 ;

%minimize
[alpha_star,fval] = fminsearch(fun,alpha_0);

V_DRO(t_s, i_ex) = fval;

%alternative cost for small radius

V_DRO_alternative(t_s, i_ex) = sum(c_vec.*p_star);    



%compute disappointments
%===========================

%IPS
if V_e >= V_IPS(t_s, i_ex)
    disap_IPS(t_s, i_ex) = 1;
else
    disap_IPS(t_s, i_ex) = 0;
end

%Beta-IPS
if V_e >= V_IPS_beta(t_s, i_ex)
    disap_IPS_beta(t_s, i_ex) = 1;
else
    disap_IPS_beta(t_s, i_ex) = 0;
end

%DRO
if V_e >= V_DRO(t_s, i_ex)
    disap_DRO(t_s, i_ex) = 1;
else
    disap_DRO(t_s, i_ex) = 0;
end


end

end

DataMatrix = V_DRO';
DataMatrix = V_IPS';
DataMatrix = V_IPS_beta';
save('off_policy_experiment_data')



%% PLOTS

%disappointment plots
for ts=1:length(T_vec)
    di_IPS(ts) = sum(disap_IPS(ts,:))/N_ex;
    di_IPS_beta(ts) = sum(disap_IPS_beta(ts,:))/N_ex;
    di_DRO(ts) = sum(disap_DRO(ts,:))/N_ex;
end

figure
title('disappointment')
plot(T_vec,di_IPS,'blue')
hold on
plot(T_vec,di_IPS_beta,'cyan')
plot(T_vec,di_DRO,'red')
legend('IPS','IPS_beta','DRO')


figure
hold on
title('Value')
for i=1:N_ex
    plot(T_vec,V_IPS(:,i),'b x')
    plot(T_vec,V_IPS_beta(:,i),'c x')
    plot(T_vec,V_DRO(:,i),'r x')
    plot(T_vec,ones(length(T_vec),1)*V_e,'g')
end
legend('IPS','beta-IPS','DRO','true cost')

