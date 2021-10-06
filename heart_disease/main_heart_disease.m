%Hearth disease problem:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Tabula rasa
clear all 
close all
clc


%parameters
N_train = 20; %size of biased training data-set
N = N_train;

%load data
T = readtable('heart.csv');

%label (disease or not) make it -1, 1
Y(:,1) = T.target;
Y(:,1) = 2*Y(:,1)-ones(length(Y(:,1)),1);

%features
X(:,1) = T.sex;
X(:,2) = T.age;
X(:,3) = T.cp;
X(:,4) = T.trestbps;
X(:,5) = T.chol;
%X(:,6) = T.fbs;


%normalize features
 for i=1:length(X(1,:))
     X(:,i) = X(:,i)/max(X(:,i));
 end



average_values_X = mean(X);

d = length(X(1,:)) + 1; %dimension of features and labels
N_data = length(Y);

%% generation of biased data set 
j=1;
for i=1:length(Y)
    if  X(i,2)>0.83 && X(i,1)==1  
        X_test_pot(j,:) = X(i,:);
        Y_test_pot(j,:) = Y(i,1);
        j = j+1;
    end
end

N_select = length(X_test_pot(:,1));

%select N_train out of N_select randomly
ind_train = randsample(N_select,N_train);

X_train = X_test_pot(ind_train,:);
Y_train = Y_test_pot(ind_train);

average_values_X
mean(Y)

mean(X_train)
mean(Y_train)

%prior knowledge encoded by the set E
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta = 0.01;



E = [(1-eta)*average_values_X', (1+eta)*average_values_X';
   (1-eta)*mean(Y), (1+eta)*mean(Y)];





%% Simulation parameters

% DESIRED ACCURACY
eps = 10^(-3);


% FIND A SLATER POINT USING SOS PROGRAMMING
%C = slater_point(E);
C = 0.1; %to be found !!

%theorem parameters
D = 0.5*sqrt(norm(E(:,2)));

delta = 0.1;    %to be found !!
 
%smoothing parameters
eta_1 = eps/(4*D);
eta_2 = (eps*delta^2)/(2*C^2);


eta = [eta_1,eta_2];

% operator bound
alpha = 1;    %approximation


%Lipschitz constant
L =  1/eta_1 + eta_2 + D;

%number of iterations
N1 = 2*sqrt( (8*D*C^2)/(eps^2*delta^2) + (2*alpha^2*C^2)/(eps*delta^2) +1 )*log( (10*(eps +2*C))/eps);
N2 = 2*sqrt( (8*D*C^2)/(eps^2*delta^2) + (2*alpha^2*C^2)/(eps*delta^2) +1 )*log(C/(eps*delta*(2-sqrt(3)))*sqrt( 4*(4*D/eps + alpha^2 +eps*delta^2/(2*C^2))*(C+eps/2)));                          

N_iter = ceil(max(N1,N2))


%loss function

loss = @(beta,x,y) log( 1+exp(-y* (beta*x')) );




% different DRO radii to simulate
r_sim = logspace(-6,1,15);





% START SIMULATION

 for j=1:length(r_sim) %loop over different radii
    r = r_sim(j)
    
    %Read out training data
    xi = [X_train';Y_train'];
    %format xi = [ [x_1;y_1], [x_2;y_2], ..., [x_N;y_N] ]
    
    
%% Algorithm


%start values
w=zeros(N_iter,d);
yy=zeros(N_iter,d);

for k=1:N_iter-1
    yy(k+1,:) = w(k,:) + 1/L*gradient_eval(w(k,:),eta,N,E,d,xi);
    w(k+1,:) = yy(k+1,:)+ (sqrt(L) - sqrt(eta(2)))/(sqrt(L) + sqrt(eta(2))) * (yy(k+1,:)-yy(k,:));
end
    


%% recompute max_ent moments - check feasibility of computed \widehat P_N^\star
z = yy(end,:);


%1. start with x-domain

x_tr_data = xi(1:d-1,:);
y_tr_data = xi(end:end,:);


% NEED A STABLE IMPLEMENTATION
gamma = zeros(1,length(x_tr_data));


for jj=1:length(x_tr_data)
    be(jj) = 0;
    for i=1:d
      be(jj) = be(jj) - z(i)*xi(i,jj);
    end
    al(:,jj) = xi(:,jj);
end
max_be = max(be);
for jj=1:N
    ga(jj) = be(jj) - max_be;
end



%compute log of fraction
buuf = zeros(length(al(:,1)),1);
buuf1 = zeros(length(al(:,1)),1);
buuf2 = 0;
for jj=1:N
    buuf = buuf + al(:,jj) * exp(ga(jj));
    buuf1(jj) =  exp(ga(jj));
    buuf2 = buuf2 + exp(ga(jj));
end



Log_nom = max_be + log(buuf);
Log_nom1 = max_be + log(buuf1);
Log_denom = max_be + log(buuf2);


for jj=1:length(x_tr_data)
    gamma(jj) = exp(Log_nom1(jj) - Log_denom);
end
moments_max_ent = exp(Log_nom - Log_denom);


        
%% Build optimal predictor (requires solving an optimization problem) 

%parameters
param1.x_data = x_tr_data;
param1.y_data = y_tr_data;
param1.gamma = gamma;
param1.r = r;

%for best-in-class comparison
param1.x_data_tot = X';
param1.y_data_tot = Y';





%function for fminsearch
f = @(z) predictor_function(z,param1);

f_SAA = @(z) predictor_function_SAA(z,param1); 

%starting point
alpha_0 = 10;
beta_0 = 2*ones(1,length(x_tr_data(:,1)));
z0 = [alpha_0 beta_0];

%z0 = alpha_0;

%minimize
[z,fval] = fminsearch(f,z0);




N_grid = 25;
xx_grid = linspace(-15,40,N_grid);
buf_min = 10^6;
for i0=1:N_grid
    for i1=1:N_grid
        for i2 = 1:N_grid
            for i3=1:N_grid
                for i4=1:N_grid
                    for i5=1:N_grid
                        alpha_t = xx_grid(i0);
                        beta_t = [xx_grid(i1), xx_grid(i2), xx_grid(i3), xx_grid(i4), xx_grid(i5)];
                        val_temp = f([alpha_t,beta_t]);
                        if val_temp < buf_min
                            buf_min = val_temp;
                            alpha_save = alpha_t;
                            beta_save = beta_t;
                        end
                    end
                end
            end
        end
    end
end



disp('gridding-Approach')
min_MC(j) = buf_min

    
%save optimal values 
alpha(j) = alpha_save;
beta(j,:) = beta_save';
J_N(j) = buf_min;  %certificate





       
%minimize
[z_SAA,fval_SAA] = fminsearch(f_SAA,z0);


disp('c_r predictors - SAA')
fval 
fval_SAA 





 
J_N_SAA = fval_SAA;
    


%% Test this beta-classifies on the entire dataset

false_counter(j) = 0;
false_counter_train(j)=0;



% DRO method
for i=1:N_train
    if (1+ exp(-beta(j,:)*X_train(i,:)'))^(-1)>0.5
        Y_pred(j,i) = 1;
    else
        Y_pred(j,i) = -1;
    end
    
    if Y_pred(j,i) ~= Y_train(i) 
        false_counter_train(j) = false_counter_train(j) + 1;
    end
end




for i=1:N_data
    if (1+ exp(-beta(j,:)*X(i,:)'))^(-1)>0.5
        Y_pred(j,i) = 1;
    else
        Y_pred(j,i) = -1;
    end
    
    if Y_pred(j,i) ~= Y(i) 
        false_counter(j) = false_counter(j) + 1;
    end
end


%out-of-sample-cost
buf_ooc = 0;
for i=1:N_data
    buf_occ = buf_ooc + loss(beta(j,:),X(i,:),Y(i));
end
ooc_DRO(j) = 1/N_data*buf_occ



 end

%% Vanilla logistic regression
 
 %function for fminsearch
g = @(z) vanilla_LR(z,param1);

%starting point
alpha_0 = 1;
beta_0 = zeros(1,length(x_tr_data(:,1)));
z0 = [alpha_0 beta_0];

%minimize
[z_LR,fval_LR] = fminsearch(g,z0);

%output optimal values
alpha_LR = z_LR(1);
beta_LR(:) = z_LR(2:end);
J_N_LR = fval_LR;  %certificate 

%out-of-sample-cost
buf_ooc_LR = 0;
for i=1:N_data
    buf_occ_LR = buf_ooc_LR + loss(beta_LR,X(i,:),Y(i));
end
ooc_DRO_LR = 1/N_data*buf_occ_LR;
 
%classification errors 
 false_counter_LR = 0;
false_counter_train_LR=0;
 
  
for i=1:N_train
    if (1+ exp(-beta_LR*X_train(i,:)'))^(-1)>0.5
        Y_pred_LR(i) = 1;
    else
        Y_pred_LR(i) = -1;
    end
    
    if Y_pred_LR(i) ~= Y_train(i) 
        false_counter_train_LR = false_counter_train_LR + 1;
    end
end




for i=1:N_data
    if (1+ exp(-beta_LR*X(i,:)'))^(-1)>0.5
        Y_pred_LR(i) = 1;
    else
        Y_pred_LR(i) = -1;
    end
    
    if Y_pred_LR(i) ~= Y(i) 
        false_counter_LR = false_counter_LR + 1;
    end
end



%% best in class logistic regression - assuming we know the entire testing set
 %function for fminsearch
g_best = @(z) vanilla_LR_best(z,param1);

%starting point
alpha_0 = 1;
beta_0 = zeros(1,length(x_tr_data(:,1)));
z0 = [alpha_0 beta_0];

% %minimize
[z_LR_best,J_N_LR_best] = fminsearch(g_best,z0);
alpha_LR_best = z_LR_best(1);
beta_LR_best = z_LR_best(2:end);

%out-of-sample-cost
buf_ooc_LR_best = 0;
for i=1:N_data
    buf_occ_LR_best = buf_ooc_LR_best + loss(beta_LR_best,X(i,:),Y(i));
end
ooc_DRO_LR_best = 1/N_data*buf_occ_LR_best;


 
%classification errors 
 false_counter_LR_best = 0;
false_counter_train_LR_best=0;







for i=1:N_data
    if (1+ exp(-beta_LR_best*X(i,:)'))^(-1)>0.5
        Y_pred_LR_best(i) = 1;
    else
        Y_pred_LR_best(i) = -1;
    end
    
    if Y_pred_LR_best(i) ~= Y(i) 
        false_counter_LR_best = false_counter_LR_best + 1;
    end
end





%% output

disp('errors DRO')
 error_rate_train = false_counter_train/N_train   
 error_rate = false_counter/N_data  

 
 disp('errors vanilla LR')
 error_rate_train_LR = false_counter_train_LR/N_train   
 error_rate_LR = false_counter_LR/N_data  
 
 
  disp('capacity - best in class LR')
 error_rate_LR_best = false_counter_LR_best/N_data  
 
 
figure
semilogx(r_sim,error_rate)
hold on
semilogx(r_sim, ones(1,length(r_sim))*error_rate_LR_best)
semilogx(r_sim, ones(1,length(r_sim))*error_rate_LR)
xlabel('radius')
title('Missclassification rate')
legend('DRO method','Full information logistic regression','Naiive logistic regression')


figure
semilogx(r_sim,ooc_DRO)
hold on
semilogx(r_sim,ones(1,length(r_sim))*ooc_DRO_LR_best)
semilogx(r_sim,ones(1,length(r_sim))*ooc_DRO_LR)
xlabel('radius')
title('Out-of-sample cost')
legend('DRO method','Full information logistic regression','Naiive logistic regression')

figure
semilogx(r_sim,J_N)
xlabel('radius')
title('Upper confidence bound')


%%

save('data_hearth_high_dim_3')
