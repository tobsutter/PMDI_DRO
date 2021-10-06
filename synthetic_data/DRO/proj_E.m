function res = proj_E(z,E)
%compute proj_E(z)


%format for E: E=[Emin Emax; Emin Emax; .... , Emin Emax];


for i=1:length(z)
    Emin = E(i,1);
    Emax = E(i,2);
    
    res(i)=z(i);
    if z(i) < Emin
        res(i) = Emin;
    end
    if z(i) > Emax
        res(i) = Emax;
    end
end

