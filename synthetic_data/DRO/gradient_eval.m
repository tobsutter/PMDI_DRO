function val = gradient_eval(z,eta,N,E,d,xi)


%inputs: eta,z, N, E, d, xi


%format xi = [ [x_1;y_1], [x_2;y_2], ..., [x_N;y_N] ]


%give back G_eta(z)

buf_nom = 0;
buf_denom = 0;
buf = 0;

for j=1:N
    buf = 0;
    for i=1:d
      buf = buf - z(i)*xi(i,j);
    end
    buf_nom = buf_nom + xi(:,j)*exp(buf);
    buf_denom = buf_denom + exp(buf);
end


G_eta = - proj_E(z./eta(1),E) - eta(2)*z + (buf_nom./buf_denom)';



val = G_eta;








