function res = Q(xp,x,a,param)


t = param.t;

if xp == 0
    res = (1-t)^(x+a);
else
    if x+a >= xp
        res = (1-t)^(x+a-xp)*t;
    else
        res = 0;
    end
end

end

