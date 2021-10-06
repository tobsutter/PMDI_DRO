function val = vanilla_LR(z,param)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

x_data = param.x_data;
y_data = param.y_data;

alpha = z(1);
beta = z(2:end);

buf = 0;
for i=1:length(x_data)
buf = buf + log(1+ exp(-y_data(i)*(beta*x_data(:,i))));
end

val = buf/length(x_data);



end

