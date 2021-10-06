function [res] = p_te(x)
% test distribution

buf = 0;
for i=1:length(x)
    buf = buf+x_i;
end

res = 2/length(x)*buf;

end

