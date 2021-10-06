function val = predictor_function_IWERM(z,param)


x_data = param.x_data;
y_data = param.y_data;


beta = z;



%evaluate function
buf = 0;

for i=1:length(x_data)
    sum_x = 0;
    for j=1:length(z)
        sum_x = sum_x + x_data(j,i);
    end
    buf = buf+ log(1 + exp(-y_data(i)*(beta*x_data(:,i))))*2/(length(z))*sum_x;
end
buf = 1/length(x_data)*buf;


val = buf';



end

