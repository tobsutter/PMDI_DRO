function [res] = int_risk_1(beta,x,y)




loss = @(beta,x,y) log( 1+exp(-y* (beta*x)) );



res = 2/length(x).*sum(x).*loss(beta,x,y);



end

