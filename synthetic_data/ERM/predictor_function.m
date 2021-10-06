function val = predictor_function(z,param)


x_data = param.x_data;
y_data = param.y_data;


beta = z;



%evaluate function
buf = 0;

for i=1:length(x_data)
    buf = buf+ log(1 + exp(-y_data(i)*(beta*x_data(:,i))));
end
buf = 1/length(x_data)*buf;


val = buf;


end

