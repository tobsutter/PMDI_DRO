function val = predictor_function_SAA(z,param)


x_data = param.x_data;
y_data = param.y_data;
gamma = param.gamma;
r = param.r;

alpha = z(1);
beta = z(2:end)';

%beta = param.beta;
beta = beta';

buf_test = 0;

for i=1:length(x_data)    
    buf_test = buf_test + gamma(i)*log(1 + exp(-y_data(i)*(beta*x_data(:,i))));
end
%buf_test = buf_test/length(x_data);


val = buf_test;



end

