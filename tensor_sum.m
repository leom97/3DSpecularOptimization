function C = tensor_sum(A,B)
%tensor_sum C(i1,i2,...,j1,j2,...,:) A(i1,..., :) + B(j1,...,:)

    sa = size(A);
    sb = size(B);

    if length(sa) ~= length(sb)
        disp("A, B must have the same number of dimensions");
        return;
    elseif sa(end) ~= sb(end)
        disp("A and B must have equal length in their last dimension, and this should be not 1");
        return;
    end

    sam = sa(1:(end-1));
    samc = num2cell(sam);
    sbm = sb(1:(end-1));
    sbmc = num2cell(sbm);
    d = sa(end);

    A_comp = repmat(A,sbmc{:});
    B_comp = repelem(B,samc{:});
    C1 = A_comp+B_comp;

    alt = repelem(sam,2);
    alt(2:2:end) = sbm;
    C2 = reshape(C1,[alt,d]);
    
    f = 2 * (1:length(sam));
    g = 2 * (1:length(sam))-1;
    
    p = [g,f,2*length(f)+1];
    C = permute(C2,p);

end

