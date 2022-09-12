% Weight function for Cauchy robust estimator. lambda is a variance and can
% be user refined.

function w = w_fcn_spec_cauchy(r,s, lambda) 

    N = (2*lambda.^2.*s.^2.*(exp(r)-1));    % added correction term lambda^2 here
    D = r.*(lambda.^2.*exp(2*r)+s.^2.*((exp(r)-1).^2));
    w = N./D;

    % Note, lim(r->-inf, s->0) is not defined!
    % Let r -> -inf, s->0: we get the form [-2 l^2 s^2 exp(-2r)]/[r(l^2+s^2 exp(-2r))]
    % The important thing is s exp(-r)
    C = s.*exp(-r);
    % Suppose s exp(-r) is finite: we get...
    w(isfinite(C) & isnan(w)) = 0;
    % Suppose s exp(-r) is +Inf: we get
    w(C==Inf & isnan(w)) = 0;
    
%     w(C==0) = 0;    % compute the limit for r->-inf and s->0, assuming e^(-2r) s^2 -> 0 (it is lim... -2s^2e^2r/(r l^2))
%     w((abs(N)==Inf | abs(D) == Inf))=0;
%     if (abs(N)==Inf & abs(D) == Inf)
%         disp("Yet another problem");
%     end
%     w(r==0) = 2/lambda*s(r==0).^2;
%     w(s==0) = 0;
end

