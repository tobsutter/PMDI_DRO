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




T_vec = linspace(10,400,20);
T_vec = linspace(10,1000,20);



param.r = 10^(-2);    %DRO radius

N_ex = 50; % #experiments
N_ex = 50;


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
occ_e_check = zeros(n,m);
occ_e = zeros(n,m);
cost_traj = [];
for tt = 1:T_eval
    a_distr = Pi_e(x_last+1,:); %select randomized policy
    a = gendist(a_distr,1,1)-1; %draw policy realization
   % a = discretesample(a_distr,1)-1; %draw policy realization
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
                occ_e_check(i,j) = occ_e_check(i,j)+1;
            end
        end
    end
    
    
end

%ground truth
V_e_true = sum(cost_traj)/T_eval


%compute stationary distribution
for in=1:n
    prob_x_e(in) = length(find(x_traj==in-1))/T_eval;
end

    %compute occupation measure
    for i=1:n
        for j=1:m
                occ_e(i,j) = Pi_e(i,j)*prob_x_e(i);
        end
    end

occ_e = occ_e + ones(size(occ_e)).*10^(-15); %avoid singularities

occ_e_check = 1/T_eval.*occ_e_check; %empirical behavioural occupation measure 







%% compute true (unknown) behavioural frequency
T_eval = 10^5;

%simulate occupation measure
x_last=0; %initial condition
x_traj = [];
a_traj = [];
cost_traj = [];
for tt = 1:T_eval
    a_distr = Pi_b(x_last+1,:); %select randomized policy
    a = gendist(a_distr,1,1)-1; %draw policy realization
   % a = discretesample(a_distr,1)-1; %draw policy realization
    a_traj = [a_traj; a];
    w = w_noise(tt); %draw random demand    
    x_next = max(0,min(n-1,x_last+ a)-w);
    x_traj = [x_traj;x_next];
    cost_traj = [cost_traj; cost(x_next,a,param)];
    x_last = x_next; 
    
end



%compute stationary distribution
for in=1:n
    prob_x_b_true(in) = length(find(x_traj==in-1))/T_eval;
end










%% True (unknown) evaluation cost


%correction term for pi_e
Pi_e_c = zeros(n,m);
for i=1:n
    for j=1:m
        Pi_e_c(i,j) = occ_e_check(i,j)/prob_x_e(i);
    end
end

% use this corrected evaluation policy from now on ....



for ij = 1:10

c_eval = 0;

x = 0; %initial condition
a = 0;

for tt=1:T_eval
    a_distr = Pi_e_c(x+1,:); %select randomized policy
    a = gendist(a_distr,1,1)-1; %draw policy realization
 %  a = discretesample(a_distr,1)-1; %draw policy realization
    w = w_noise(tt); %draw random demand
    x = max(0,min(n-1,x+ a)-w);
    c_eval = c_eval + cost(x,a,param);
end

%disp('true (unknown) evaluation cost')
V_e(ij) = c_eval/T_eval;
end

V_e = mean(V_e)












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

%Step 1: find true occupation measure


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
   % a = discretesample(a_distr,1)-1; %draw policy realization
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



%compute stationary distribution
for in=1:n
    prob_x_b(in) = length(find(x_traj==in-1))/T_sim;
end

    %compute occupation measure
    for i=1:n
        for j=1:m
                occ_b(i,j) = Pi_b(i,j)*prob_x_b(i);
        end
    end
    


%empirical occupations measure
occ_b_check = occ + ones(size(occ)).*10^(-8); %avoid singularities

occ_b_check = 1/T_sim.*occ_b_check; %empirical behavioural occupation measure  
cost_matrix_b = cost_matrix_b./T_sim;

prob_x_b = (prob_x_b + 10^(-8)*ones(1,length(prob_x_b)))/sum(prob_x_b + 10^(-8)*ones(1,length(prob_x_b)));



%correction term for pi_b
Pi_b_c = zeros(n,m);
for i=1:n
    for j=1:m
        Pi_b_c(i,j) = occ_b_check(i,j)/prob_x_b(i);
    end
end

% use this corrected behavioural policy from now on ....


%avoid singularities
for i=1:n
    Pi_b_c(i,:) = (Pi_b_c(i,:) + ones(1,m)*10^(-4))/sum(Pi_b_c(i,:) + ones(1,m)*10^(-4));
end



Pi_b_c = Pi_b;
Pi_e_c = Pi_e;



    
%% Inverse propensity weights

we = occ_e_check./occ_b_check;
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
        buf = buf + occ_e_check(i,j)*log(occ_e_check(i,j)/occ_b_check(i,j));
        a1((i-1)*m+j) = log(occ_e_check(i,j)/occ_b_check(i,j));
        q((i-1)*m+j) = occ_b_check(i,j);
    end
end
alpha = buf;

A1(1,:) = a1;
b1 = [alpha];
A2 = A;
b2 = b;
mub = occ_b_check;

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
        p_e((i-1)*m+j) = occ_e_check(i,j);
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













%% doubly robust method [Jiang, Li 2016]
 
s_data = x_traj;
a_data = a_traj;


H=T_sim;


% 
% 
% 
% 
% %compute optimal Q (for simplicity we assume this is possible)
% %run Bellman interations
%  
% V(:,1) = zeros(n,1);
% 
% 
% 
% for h=1:H
% 
% for i=1:n
%     x = x_grid(i);
%     for j=1:m
%         a = a_grid(j);
%         %Monte-Carlo approximation to the integral
%         N_MC = 5000;
%         w_n = random('geo',param.t,1,N_MC);
%         for y=1:length(w_n)
%             x_next(y) = max(0,min(n-1,x+ a)-w_n(y));
%         end
%         for iy =1:n
%             ind = find(x_next == iy-1);
%             q_kern(iy)= 1/N_MC*length(ind);
%         end
%         buf = 0;
%         for iy=1:n
%             buf = buf + V(iy,h)*q_kern(iy);
%         end
%         Q(i,j,h) = cost(x,a,param) + buf;
%     end
% end
% 
% for i=1:n
%     x = x_grid(i);
%     buf = 0;
%     for j=1:m
%         a=a_grid(j);
%         %buf = buf+Q(x+1,a+1)*Pi_e(x+1,a+1);
%         buf = buf+Q(x+1,a+1,h)*Pi_e_c(x+1,a+1);
%     end
%     V(i,h+1) = buf;
% end
% end
% 
% 
% % test naiive value iteration:
% disp('test value iteration')
% V(:,end)/H  
% 
% 


% SOLVE LP - approach taken by the infinite horizon doubly robust paper

f = zeros(1,n);
f = [1 f]; %objective

A = -eye(n);
A = [-ones(n,1) A];
for i=1:n
    x = x_grid(i);
    buf = zeros(n,1);
    for j=1:m
        a = a_grid(j);
        %Monte-Carlo approximation to the integral
        N_MC = 5000;
        w_n = random('geo',param.t,1,N_MC);
        for y=1:length(w_n)
            x_next(y) = max(0,min(n-1,x+ a)-w_n(y));
        end
        for iy =1:n
            ind = find(x_next == iy-1);
            q_kern(iy)= 1/N_MC*length(ind);
        end
        for ix = 1:n
            buf(ix) = buf(ix) + Pi_e_c(x+1,a+1)*q_kern(ix);
        end
    end
    for ix=1:n
        A(i,ix+1) = A(i,ix+1) + buf(ix);
    end
end
   
%find b
for i=1:n
    x = x_grid(i);
    buf = 0;
    for j=1:m
        a = a_grid(j);
        buf = buf - Pi_e_c(x+1,a+1)*cost(x,a,param);
    end
    bb(i) = buf;
end
        
%solve LP
x = linprog(f,A,bb);
optimal_value = x(1)
Value_fct = x(2:end)



%generate data:multiple behavioral trajectories
%behavioual occupupation measure and cost
N_tra = 100;
x_traj_tot = x_traj;
a_traj_tot = a_traj;
w_noise_new = random('geo',param.t,1,10^6);


a_save = a_traj;

for i_t = 1:N_tra-1

x_last=0; %initial condition

x_traj = [];
a_traj = [];
occ = zeros(n,m);
cost_traj = [];
for tt = 1:T_sim
    a_distr = Pi_b(x_last+1,:); %select randomized policy
    a = gendist(a_distr,1,1)-1; %draw policy realization
   % a = discretesample(a_distr,1)-1; %draw policy realization
    a = a_save(tt);
    a_ind = find(a==a_grid);
    a_traj = [a_traj; a];
    w = w_noise_new(tt); %draw random demand
    x_next = max(0,min(n-1,x_last+ a)-w);
    x_ind = find(x_next == x_grid);
    x_traj = [x_traj;x_next];
    cost_traj = [cost_traj; cost(x_next,a,param)];
    x_last = x_next;
end

x_traj_tot = [x_traj_tot x_traj];
a_traj_tot = [a_traj_tot a_traj];
end





%continue with infinite horizon doubly robust paper
for i=1:n
    x = x_grid(i);
    for j=1:m
        a = a_grid(j);
        beta(i,j) = Pi_e_c(i,j)/Pi_b(i,j);
      % beta(i,j) = Pi_e(i,j)/Pi_b(i,j);
    end
    w_DR(i) = prob_x_e(i)/prob_x_b_true(i);
end


nom = 0;
denom = 0;
for i_t = 1:N_tra
    s_data = x_traj_tot(:,i_t);
    a_data = a_traj_tot(:,i_t);
    for t=1:H-1
        x = s_data(t);
        a = a_data(t);
        nom = nom + w_DR(x+1)*(beta(x+1,a+1)*(cost(x,a,param) + Value_fct(s_data(t+1)+1))-Value_fct(x+1));
        denom = denom + w_DR(x+1);
    end
end


R_DR(t_s, i_ex) = nom/denom;





% 
% % as a parity check - run classical DP (1. Value iteration, then Corollary
% % 5.6.6 from hernandez lerma ===> this is optimal control (different)
% 
% V_HL(:,1) = zeros(n,1);     %initialization
% for it = 1:H    %run value iteration
%     for ix = 1:n
%         x = ix-1;
%         buf_min =10^6;
%         for ia = 1:m
%             a = ia-1;
%              %Monte-Carlo approximation to the integral
%              N_MC = 5000;
%              w_n = random('geo',param.t,1,N_MC);
%             for y=1:length(w_n)
%                 x_next(y) = max(0,min(n-1,x+ a)-w_n(y));
%             end
%             for iy =1:n
%                 ind = find(x_next == iy-1);
%                 q_kern(iy)= 1/N_MC*length(ind);
%             end
%             buf = 0;
%             for iy=1:n
%                 buf = buf + V_HL(iy,it)*q_kern(iy);
%             end
%             %find minimum in bellman
%             V_buf = cost(x,a,param) + buf;
%             if V_buf < buf_min
%                 buf_min = V_buf;
%                 V_HL(ix,it+1) = V_buf;
%             end
%         end
%     end
% end
%         
%  V_HL_final = V_HL(:,end)/H       
%   





% for tt=1:H
%     x_ind = s_data(tt)+1;
%     a_ind = a_data(tt)+1;
%     
%     disp('r_t + V_t - Q(s_t,a_t)')
%     cost_traj(tt) + V_DR(tt)-Q(x_ind,a_ind,tt)
%     
%     V_DR(tt+1) = V_hat(x_ind) + rho(tt)*(cost_traj(tt) + V_DR(tt)-Q(x_ind,a_ind,tt));
%    %V_DR(tt+1) = V_hat(x_ind) + rho(tt)*(cost_traj(tt) + V_DR(tt)-Q(x_ind,a_ind) - V_b);
%     
%     buf_fin = buf_fin + V_DR(tt+1);
%     
% end





%compute disappointments
%===========================

%IPS
if V_e_true >= V_IPS(t_s, i_ex)
    disap_IPS(t_s, i_ex) = 1;
else
    disap_IPS(t_s, i_ex) = 0;
end

%Beta-IPS
if V_e_true >= V_IPS_beta(t_s, i_ex)
    disap_IPS_beta(t_s, i_ex) = 1;
else
    disap_IPS_beta(t_s, i_ex) = 0;
end

%DRO
if V_e_true >= V_DRO(t_s, i_ex)
    disap_DRO(t_s, i_ex) = 1;
else
    disap_DRO(t_s, i_ex) = 0;
end

%DR 
if V_e_true >= R_DR(t_s, i_ex)
    disap_DR(t_s, i_ex) = 1;
else
    disap_DR(t_s, i_ex) = 0;
end










end

end

DataMatrix = V_DRO';
DataMatrix = V_IPS';
DataMatrix = V_IPS_beta';
DataMatrix = R_DR';
save('off_policy_experiment_data')










%% PLOTS

%disappointment plots
for ts=1:length(T_vec)
    di_IPS(ts) = sum(disap_IPS(ts,:))/N_ex;
    di_IPS_beta(ts) = sum(disap_IPS_beta(ts,:))/N_ex;
    di_DRO(ts) = sum(disap_DRO(ts,:))/N_ex;
    di_DR(ts) = sum(disap_DR(ts,:))/N_ex;
end

figure
title(['r=',num2str(param.r)])
plot(T_vec,di_IPS,'blue')
hold on
plot(T_vec,di_IPS_beta,'cyan')
plot(T_vec,di_DRO,'green')
plot(T_vec,di_DR,'magenta')
legend('IWERM','capped IWERM','MDI-DRO','doubly robust method')
xlabel('N')
ylabel('disappointment')



figure
hold on
title(['r=',num2str(param.r)])
for i=1:N_ex
    plot(T_vec,V_IPS(:,i),'b x')
    plot(T_vec,V_IPS_beta(:,i),'c x')
    plot(T_vec,V_DRO(:,i),'g x')
    plot(T_vec,ones(length(T_vec),1)*V_e_true,'r')
end
legend('IWERM','capped IWERM','MDI-DRO','ground truth')
xlabel('N')
ylabel('R(\theta^*_N,P_{te})')




figure
hold on
title(['r=',num2str(param.r)])
for i=1:N_ex
    plot(T_vec,V_DRO(:,i),'g x')
    plot(T_vec,ones(length(T_vec),1)*V_e_true,'r')
    plot(T_vec,R_DR(:,i),'m o')
end
legend('MDI-DRO','ground truth','doubly robust method')
xlabel('N')
ylabel('R(\theta^*_N,P_{te})')



save('OPE_sim')

