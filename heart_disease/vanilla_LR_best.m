function val = vanilla_LR_best(z,param)
%assume that we know the entire data set

x_data = param.x_data_tot;
y_data = param.y_data_tot;

alpha = z(1);
beta = z(2:end);

buf = 0;
for i=1:length(x_data)
buf = buf + log(1+ exp(-y_data(i)*(beta*x_data(:,i))));
end

val = buf/length(x_data);



end

