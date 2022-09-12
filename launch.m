clear; close all;

% Dataset loading
name = "helmet";    % or "backpack"
load("Datasets/" + name +"_non_lamb_spec_test.mat");
N = N_display;
S = S_display;

% Options
scalar_shininess = 1;   % estimate one shininess per pixel, or a global one?
do_SG = 0;  % use a spherical gaussian approximation of the BRDF model
real_max = 1;   % use a spherical gaussian approximation to x \mapsto max(.,0)
linear_rho = 1; % the specular albedo can also be optimized "linearly" and not logarithmically
maxit = 30; % max iterations for optimization run
c = 200;    % shininess initial guess
almost_zero = 0;    %  < almost_zero => truncation to 0
lambda_multiplier = 1; % control the size of a free parameter in the logarithmic estimator (like a variance)
perturbation = 5e-3;    % apply random noise of this size to the data
batch = 1;
% note: lambda_multiplier = 10 for helmet, =1 for backpack

% Initial guess
% see line 117

%Experimental options, leave unactive
enable_step_control = 0;
% Masking, it is definitely doing something
do_shininess_masking = 0;
percentile_shininess = 90;  % every intensity value below this percentile will not be considered for estimating shininess
do_rho_S_masking = 0;
rho_S_threshold = .5;
% Constrained optimization
constrain = 0;
alpha = 1e-16;   % Variable that says how much should we diminish delta_tilde in case the observations are all black

% Other stuff
nimgs = size(I,4);
nchannels = size(I,3);
mu_cos = 1; lambda_cos = 3;
spec_thresh = 0;

nrows = size(I,1);
ncols = size(I,2);

if real_max==1
    mu_cos = 1;
end

%% Some warnings
if constrain==1 && linear_rho==0
    disp("Haven't implemented this yet")
    return
end

%% Some variables, for debugging

% load("some_variables_" + name + ".mat")
% sn_lr = sn;
% clear sn;
% hn_lr = hn;
% clear hn;

%% Create specular images

Phi = vecnorm(S,2,2);
S = S./Phi;
S = repmat(S, [1,1,nchannels]);

I = permute(I,[1,2,4,3]);
Phi = repmat(Phi,[1,nchannels]);

% Compute h . n
view_vectors = (0-XYZ)./vecnorm(XYZ,2,3);   % 2 norm along direction 3
light_directions = permute(S,[1,3,2]);           % it is now n_frame x nchannels x 3
h = tensor_sum(view_vectors, light_directions);
h = h./vecnorm(h,2,length(size(h)));    % so, we index h as follows: (u,v,im,ch,:)
N_exp = permute(N, [1,2,4,5,3]);
N_exp = repmat(N_exp,[1,1,nimgs,nchannels,1]);
hn =sum(h.*N_exp,5);    % a variable u, v, im, ch 
% hn = max(2*rand(size(hn))-1,0);

% Compute S . n
sn = tensorprod(light_directions, N, 3);    % https://www.mathworks.com/matlabcentral/answers/1690105-dot-product-of-tensor
sn = permute(sn, [3,4,1,2]);


if do_SG==1
    alpha_tilde = hn - 1;
else
    alpha_tilde = real(log(hn));
end
if real_max==0
    shad_term = lambda_cos*(sn-1);    % could have absorbed Phi into eta_tilde... and no need for reshaping shenanigans, and in coherence with alpha_tilde
else
    shad_term = real(log(max(0,sn)));
end

beta_tilde = shad_term+real(log(permute(repmat(Phi,[1,1,nrows,ncols]),[3,4,1,2]))); % TODO: what if a light has 0 intensity?

%% Optimization 

% Create I_specular
s_GT = c * ones(size(I,1), size(I,2), nimgs, nchannels);
rho_S_GT = permute(repmat(rho_S, [1,1,1,nimgs]),[1,2,4,3]);
rho_GT = permute(repmat(rho, [1,1,1,nimgs]),[1,2,4,3]);
Phi = permute(repmat(Phi,[1,1,size(I,1), size(I,2)]),[3,4,1,2]);

I_S_full = rho_S_GT.*(s_GT+2).*max(0, hn).^s_GT.*max(0,sn).*Phi;
I_D_full = rho_GT.*max(0,sn).*Phi;

I_tilde_no_log = I_S_full + perturbation*(2*rand(size(I_S_full))-1);
I_tilde_no_log_all = I_tilde_no_log;
I_tilde = real(log(I_tilde_no_log)); 

spec_obs = min(squeeze(mean(mean(I_tilde_no_log,4),3)),1);

% Initial variables
shininess = 0 * ones(size(squeeze(s_GT(:,:,1,1))));
% rho_S = 1 * ones(size(squeeze(rho_S_GT(:,:,1,1))));
% rho_S = (squeeze(rho_S_GT(:,:,1,1)));
rho_S =  1e-9 + 0 * ones(size(squeeze(rho_S_GT(:,:,1,1)))); % seems very important to get this initial thing right

% Some masking variables
all_stuff = reshape((I_tilde_no_log_all),nrows,ncols,[]);
all_stuff = abs(all_stuff(mask));
max_robust = prctile(all_stuff, percentile_shininess, "all");
if do_shininess_masking==1
    mask_shininess = I_tilde_no_log_all<max_robust;
else
    mask_shininess = ones(size(I_tilde_no_log));
end

% Estimators stuff
% rmask = logical( mask .* (I_tilde_no_log > almost_zero).*(sn>almost_zero));
if real_max==0
    shad_term = exp(lambda_cos*(sn-1)).*mu_cos;
else
    shad_term = max(0,sn);
end
if do_SG==1
    E_term = exp(shininess.*(hn-1)).*(hn>0);
else
    E_term = max(0, hn).^shininess;
end
I_S_est = E_term.*Phi.* rho_S.* (shininess+2).*shad_term;
all_stuff = reshape(abs(I_tilde_no_log_all-I_S_est),nrows,ncols,[]);
% all_stuff = reshape((I_tilde_no_log_all),nrows,ncols,[]);
all_stuff = all_stuff(mask);
lambda = lambda_multiplier.*max(.15, real(log(max(all_stuff))));
% lambda = lambda_multiplier.*max(.15, max(all_stuff));

% Cauchy estimator
w_fcn_spec = @(r, s) w_fcn_spec_cauchy(r,s, lambda);
w_fcn = @(x) 2*lambda.^2./(lambda.^2+x.^2);
phi_fcn = @(x) lambda.^2*log(1+x.^2./lambda.^2);

% % Tukey estimator (not performing very well)
% lambda = 1e3;
% w_fcn_spec = @(r, s) w_fcn_spec_tukey(r,s, lambda);
% w_fcn = @(x) lambda.^2.*((abs(x)<=lambda).*(1-(1-x.^2./lambda.^2).^3) + (abs(x)>lambda));
% phi_fcn = @(x) 6*x.*(1-x.^2./lambda.^2).*((abs(x)<=lambda));


% Compute the mask, and apply it where possible, now
rmask = logical( mask .* (I_tilde_no_log > almost_zero).*(sn > almost_zero) .* (hn > almost_zero));
alpha_tilde(~rmask) = 0;
beta_tilde(~rmask) = 0;
I_tilde(~rmask) = 0;
I_tilde_no_log(~rmask) = 0;

shininess_best= shininess;
shininess_old = shininess;
rho_S_best = rho_S;
rho_S_old = rho_S;
Err_scalar = []; % Errors
insuccess = 0;
for k=1:maxit                       

    % Step size control/error computation (experimental, don't use this)
    step_shininess = (shininess-shininess_old);
    nstep_size = 1;
    for j = 1:nstep_size
        if real_max==0
            shad_term = exp(lambda_cos*(sn-1)).*mu_cos;
        else
            shad_term = max(0,sn);
        end
        if do_SG==1
            E_term = exp(shininess.*(hn-1));
        else
            E_term = hn.^shininess;
        end
        I_S_est = E_term.*Phi.* rho_S.* (shininess+2).*shad_term;
        I_S_est(~rmask) = 0;
        err_vector = (I_S_est-I_S_full);
        err_vector = err_vector(logical(rmask.*(rho_S<inf)));
        energy_linear = sum(phi_fcn(err_vector))/numel(rmask(rmask>0));
        overestimation = sum(err_vector> 0, "all")/sum(err_vector~=0, "all");
        err_scalar = norm(err_vector, 'fro')/numel(rmask(rmask>0));
        if j==1
            Err_scalar = [Err_scalar,err_scalar]; %#ok<AGROW> 
        else
            Err_scalar(end) = err_scalar;
        end

        % Don't do step size control at the beginning
        if k==1 || enable_step_control==0
            break;
        end
        if scalar_shininess==0
            disp("Implement me");
            pause;
        end

        % At this point we have shininess old different frmo shininess new,
        % and also two different values in Err_scalar

        sign_step = sign(Err_scalar(end-1) - Err_scalar(end));
        if sign_step>=0
            break;
            % We have accepted the step
        end
        
        % If we decrease the energy, reduce step size
        if j<nstep_size
            shininess = shininess_old + step_shininess/2^j;
        else
            % Don't accept anything
            shininess = shininess_old;
            rho_S = rho_S_old;
            if real_max==0
                shad_term = exp(lambda_cos*(sn-1)).*mu_cos;
            else
                shad_term = max(0,sn);
            end
            if do_SG==1
                E_term = exp(shininess.*(hn-1));
            else
                E_term = hn.^shininess;
            end
            I_S_est = E_term.*Phi.* rho_S.* (shininess+2).*shad_term;
            I_S_est(~rmask) = 0;
            err_scalar = norm((I_S_est-I_S_full).*rmask, 'fro')/numel(rmask(rmask>0));
            Err_scalar(end) = err_scalar;
            insuccess = 1;
        end
    end

    if err_scalar <= min(Err_scalar)
        shininess_best = shininess;
        rho_S_best = rho_S;
    end
    
    output = "It: " + num2str(k-1) + ", shininess mean: " + num2str(mean(shininess(mask))) +  ", log(e): " + num2str(log(err_scalar)) + ", mean rho_S: " + num2str(mean(rho_S(mask),'all')) + ", % overestimation: " + num2str(overestimation);
    disp(output) 
    shininess_old = shininess;
    rho_S_old = rho_S;

    if insuccess == 1
        break;
    end

    % Compute the interesting variables
    if 1>0
        eta_tilde_pixels = (shininess+2).*rho_S.*mu_cos;  % (u,v), full storage
        eta_tilde = repmat(eta_tilde_pixels, [1,1,nimgs, nchannels]);   % also (u,v,nimgs,nchannels)
        eta_tilde = real(log(eta_tilde));

        delta_tilde = rho_S.* (shininess+2);    % to be used only for do_linear = 1. It is here as it shouldn't be computed with the new shininess
        
        shininess_tilde = repmat(shininess, [1,1,nimgs, nchannels]);
    
        % Where there are NaNs or logarithm of negative values, just put 0
        % Wrong!
        shininess_tilde(~rmask) = 0;
        eta_tilde(~rmask) = 0;
        % Done once, out of the loop
%         alpha_tilde(~rmask) = 0;
%         beta_tilde(~rmask) = 0;
%         I_tilde(~rmask) = 0;
%         I_tilde_no_log(~rmask) = 0;
    end
        
    % Updating eta_tilde/delta_tilde
    % Do rho_S masking
    if do_rho_S_masking == 1
        rho_S_mask = (I_S_est ./ (E_term.*Phi.* (shininess+2).*shad_term))<(1 + rho_S_threshold);
    else 
        rho_S_mask = ones(size(rmask));
    end
    if linear_rho==0
        % Residuals, weights
        res_spec = (I_tilde - beta_tilde) - ...
            (eta_tilde + shininess_tilde .* alpha_tilde);
        w_eta_tilde = w_fcn_spec(res_spec, I_tilde_no_log);
        w_eta_tilde(~(rmask & rho_S_mask)) = 0;
        if isnan(sum(w_eta_tilde,'all')) || abs(sum(w_eta_tilde,'all')) == Inf
            disp('Error on the weights...')
            pause
        end
        % Least squares with handcrafted estimator
        w_eta_tilde = reshape(w_eta_tilde,size(w_eta_tilde,1), size(w_eta_tilde,2), []);    % reshape into (u,v ...)
        denom_eta_tilde = sum(w_eta_tilde, n_dim(w_eta_tilde)); % note: for (u,v), if a single weight is valid, we'll consdider him
        num_eta_tilde = (I_tilde - beta_tilde) - (shininess_tilde .* alpha_tilde);
        num_eta_tilde(~rmask)=0;
        num_eta_tilde(w_eta_tilde==0)=0;
        num_eta_tilde  =  reshape(num_eta_tilde,size(num_eta_tilde,1), size(num_eta_tilde,2), []);
        num_eta_tilde = sum(num_eta_tilde.*w_eta_tilde, n_dim(num_eta_tilde));
        eta_update_mask = denom_eta_tilde>0;
        upd_eta_tilde = num_eta_tilde(eta_update_mask)./denom_eta_tilde(eta_update_mask);

        eta_tilde_pixels(eta_update_mask) = upd_eta_tilde;

        % Updating rho_S
        rho_S(eta_update_mask) = exp(eta_tilde_pixels(eta_update_mask))./((shininess(eta_update_mask)+2)*mu_cos);  % No need to clamp to 0, shininess is already > 0...
        rho_S(mask==0) = 0;
    else
        % Get residual
        if real_max==0
            shad = exp(lambda_cos*(sn-1)).*Phi .* mu_cos;
        else
            shad = Phi.*max(0,sn);
        end
        if do_SG==1         % It is good to update E_term!
            E_term = exp(shininess.*(hn-1));
        else
            E_term = max(0, hn).^shininess;
        end
        E_term  = E_term.*shad;
        I_S_est = E_term.* delta_tilde; % shininess is updated in E_term, not in delta_tilde, correct
        res_spec = I_tilde_no_log - I_S_est;    % It is wrong to use _all!

        % Get weights (note, within the mask, I'm considering everyhting!
        w_delta_tilde = w_fcn(res_spec); % now I have masked this
        w_delta_tilde(~(rmask&rho_S_mask)) = 0;    

        % Least squares update (remember to mask it, to make it efficient)
        num_delta_tilde = w_delta_tilde.*I_tilde_no_log_all.*E_term;
        num_delta_tilde(~rmask)=0;
        num_delta_tilde(w_delta_tilde==0)=0;
        num_delta_tilde = reshape(num_delta_tilde, nrows, ncols, []);
        num_delta_tilde = sum(num_delta_tilde,3);
        denom_delta_tilde = w_delta_tilde.*E_term.^2;
        if constrain==1
            denom_delta_tilde = denom_delta_tilde + alpha * abs(log(spec_obs));
        end
        denom_delta_tilde(w_delta_tilde==0) = 0;
        denom_delta_tilde = reshape(denom_delta_tilde, nrows, ncols, []);
        denom_delta_tilde = sum(denom_delta_tilde,3);
            
        delta_update_mask = logical((denom_delta_tilde>0).*mask);
        delta_tilde(delta_update_mask) = max(num_delta_tilde(delta_update_mask)./denom_delta_tilde(delta_update_mask), 0);

        % Update rho_S
        rho_S(delta_update_mask) = delta_tilde(delta_update_mask)./(shininess(delta_update_mask)+2);
        rho_S(mask==0) = 0;
    end
    if batch == 0   
        % Update eta_tilde, if batch = 0. Otherwise, don't lag this
        % variable forward!
        eta_tilde_pixels = (shininess+2).*rho_S.*mu_cos;  % (u,v), full storage
        eta_tilde = repmat(eta_tilde_pixels, [1,1,nimgs, nchannels]);   % also (u,v,nimgs,nchannels)
        eta_tilde = real(log(eta_tilde));
    end


    % Shininess update
    if 1>0
        % Residuals, weights
        res_spec = (I_tilde - beta_tilde) - ...
            (eta_tilde + shininess_tilde .* alpha_tilde);
        w_shininess_tilde = real(w_fcn_spec(res_spec, I_tilde_no_log));
        w_shininess_tilde(~(rmask & mask_shininess)) = 0;
        if isnan(sum(w_shininess_tilde,'all')) || abs(sum(w_shininess_tilde,'all')) == Inf
            disp('Error on the weights...')
            pause
        end
        % Least squares with handcrafted estimator
        w_shininess_tilde = reshape(w_shininess_tilde,size(w_shininess_tilde,1), size(w_shininess_tilde,2), []);
        alpha_tilde = reshape(alpha_tilde, size(alpha_tilde,1), size(alpha_tilde,2), []);   % back to a pixel map
        denom_shininess_tilde = w_shininess_tilde.*alpha_tilde.^2;
        denom_shininess_tilde(w_shininess_tilde==0) = 0;
        denom_shininess_tilde = sum(denom_shininess_tilde, n_dim(w_shininess_tilde));
        if scalar_shininess==1
            denom_shininess_tilde = sum(denom_shininess_tilde,'all');
        end
        num_shininess_tilde = (I_tilde - beta_tilde) - eta_tilde;
        num_shininess_tilde(w_shininess_tilde==0) = 0;
        num_shininess_tilde = reshape(num_shininess_tilde, size(num_shininess_tilde,1), size(num_shininess_tilde,2),[]);
        num_shininess_tilde = sum(num_shininess_tilde.*w_shininess_tilde.*alpha_tilde, n_dim(num_shininess_tilde));
        if scalar_shininess==1
            num_shininess_tilde = sum(num_shininess_tilde,'all');
        end
        shininess_update_mask = denom_shininess_tilde>0;
        upd_shininess_tilde = max(num_shininess_tilde(shininess_update_mask)./denom_shininess_tilde(shininess_update_mask), 0);
        if scalar_shininess==1
            shininess_tilde = mean(shininess,'all');
        else
            shininess_tilde = shininess;
        end
        shininess_tilde(shininess_update_mask) = upd_shininess_tilde;
        if scalar_shininess==1
            shininess_tilde = ones(size(shininess)) * shininess_tilde;
        end
        % Fix back the variables
        shininess = shininess_tilde;
        shininess_tilde = repmat(shininess, [1,1,nimgs, nchannels]);
        shininess_tilde(~rmask) = 0;
        alpha_tilde = reshape(alpha_tilde, nrows, ncols, nimgs, nchannels); 
    end
end

if err_scalar <= min(Err_scalar)
    shininess_best = shininess;
    rho_S_best = rho_S;
end
output = "It: " + num2str(k) + ", shininess mean: " + num2str(mean(shininess(mask))) +  ", log(e): " + num2str(log(err_scalar)) + ", mean rho_S: " + num2str(mean(rho_S(mask),'all'));
disp(output)
Err_scalar = [Err_scalar,err_scalar];     

rho_S = rho_S_best;
shininess = shininess_best;
if real_max==0
    shad_term = exp(lambda_cos*(sn-1)).*mu_cos;
else
    shad_term = max(0,sn);
end
if do_SG==1
    E_term = exp(shininess.*(hn-1)).*(hn>0);
else
    E_term = max(0, hn).^shininess;
end
I_S_est = E_term.*Phi.* rho_S.* (shininess+2).*shad_term;

%% Errors

plot(log(Err_scalar));
e_r = abs(rho_S-rho_S_GT);
e_s = abs(c-shininess);
output = "Mean error rho_S: " + num2str(mean(e_r(mask))) + ", median: " + num2str(median(e_r(mask)));
disp(output)
output = "Mean % error shininess: " + num2str(mean(e_s(mask))/c) + ", median %: " + num2str(median(e_s(mask))/c);
disp(output)


% Get diffuse images
for i = 1:nimgs
    close all;
    subplot(1,4,1)
    imshow(squeeze(I_S_full(:,:,i,:)+I_D_full(:,:,i,:)-I_S_est(:,:,i,:)))
    title("Diffuse image estimation")
    subplot(1,4,2)
    imshow(squeeze(I_D_full(:,:,i,:)))
    title("Diffuse image ground truth")
    subplot(1,4,3)
    imshow(squeeze(I_S_full(:,:,i,:)+I_D_full(:,:,i,:)))
    title("Image, ground truth")
    subplot(1,4,4)
    imshow(squeeze(I_tilde_no_log_all(:,:,i,:)))
    title("Data from which the specular image is estimated")
    zlim([0,100])
    pause
end