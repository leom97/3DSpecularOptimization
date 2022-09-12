function [XYZ,N,rho,Phi,S,ambient,mask,tab_nrj, I_S_full] = robust_ps_V2(data,calib,params)
%robust_ps_V2 solves photometric stereo under directional sources
%
%	=== USAGE ===	
%
%	XYZ = EXPORT_OBJ(DATA,CALIB) estimates an
%	NROWS x NCOLS x 3 gridded point cloud, using the images in DATA and
%	the calibration parameters in CALIB
%
%	XYZ = EXPORT_OBJ(DATA,CALIB,PARAMS) uses the algorithm parameters in
%	set in PARAMS
%
%	[XYZ,N] = EXPORT_OBJ(...) also provides an NROWS x NCOLS x 3 normal
%	map
%
%	[XYZ,N,rho] = EXPORT_OBJ(...) also provides an 
%	NROWS x NCOLS x NCHANNELS albedo map, where NCHANNELS = 1 if the
%	images are graylevel images, and NCHANNELS = 3 if they are RGB
%
%	[XYZ,N,rho,Phi] = EXPORT_OBJ(...) also provides an NIMGS x NCHANNELS
%	matrix containing lighting intensities
%
%	[XYZ,N,rho,Phi,S] = EXPORT_OBJ(...) also provides the lights
%
%	[XYZ,N,rho,Phi,S,ambient] = EXPORT_OBJ(...) also provides also provides an ambient light map
%
%	[XYZ,N,rho,Phi,S,ambient,mask] = EXPORT_OBJ(...) also provides  the binary mask
%
%	[XYZ,N,rho,Phi,S,ambient,mask,tab_nrj] = EXPORT_OBJ(...)  also provides a vector tab_nrj
%	containing the energy at each iteration
%
%	=== DATA ===
%	
%	- Required fields:
%		- DATA.I:	NROWS x NCOLS x NIMGS (graylevel images) or
%				NROWS x NCOLS x NCHANNELS x NIMGS (color images)
%	- Optional fields:
%		- DATA.mask:	M x N binary mask (default: ones(NROWS,NCOLS))
%		- DATA.ambient:	NROWS x NCOLS (graylevel ambient image) or
%				NROWS x NCOLS x NCHANNELS (color ambient image)
%
%	=== CALIB ===
%	- Required fields:
%		- CALIB.S:	NIMGS x 3 x NCHANNELS light sources
%				or  NIMGS x 3 (assume light direction independent from color)
%
%	- Optional fields:
%		- CALIB.K:	3 x 3 intrinsics matrix
%				(default: [], i.e. orthographic)
%		- CALIB.Phi:	NIMGS x NCHANNELS intensities 
%				(default: squeeze(sqrt(sum((data.S).^2,2))))
%
%	=== PARAMETERS ===
%	- Optional fields:
%		- PARAMS.estimator: string indicating which estimator is used:
%		estimator = 'LS' uses least-squares
%		estimator = 'Cauchy' uses Cauchy's M-estimator
%		estimator = 'GM' uses Geman and McClure's M-estimator
%		estimator = 'Tukey' uses Tukey's biweight M-estimator
%		estimator = 'Welsh' uses Welsh's M-estimator
%			(default: 'LS')
%		- PARAMS.lambda: parameter of the robust estimator
%			(default: 0, since LS estimator requires no estimator)
%			(we recommend e.g. 0.1 for Cauchy, 0.7 for Lp)
%		- PARAMS.self_shadows: binary variable indicating if
%		self-shadows are included in the model or not	
%			(default: 1)
%		- PARAMS.used:	NROWS x NCOLS x NIMGS (graylevel) or
%				NROWS x NCOLS x NCHANNELS x NIMGS (color) 
%						(binary map indicating whether data is used or not)
%
%		- PARAMS.semi_calibrated: binary variable indicating if
%		intensities are refined or not
%			(default: 0)
%		- PARAMS.uncalibrated: binary variable indicating if
%		lightings are refined or not
%			(default: 0)
%		- PARAMS.do_ambient: binary variable indicating if ambient light 
%		estimation is carried out or not (EXPERIMENTAL)
%			(default: 0)
%		- PARAMS.do_albedo: binary variable indicating if albedo 
%		estimation is carried out or not
%			(default: 1)
%		- PARAMS.do_depth: binary variable indicating if depth 
%		estimation is carried out or not
%			(default: 1)
%
%		- PARAMS.z0: NROWS x NCOLS initial depth map
%			(default: 700*ones(NROWS,NCOLS))
%		- PARAMS.prior: prior parameter enforcing similarity to z0
%			(default: 1e-6)
%		- PARAMS.stabilizer: stabilizer for the descent in the lighting
%			(default: 0.1)
%
%		- PARAMS.scales: the number of scales in the multi-scale approach
%			(default: 1)
%
%		- PARAMS.maxit: max number of iterations
%			(default: 100)
%		- PARAMS.tol: relative stopping criterion on the energy
%			(default: 1e-3)
%		- PARAMS.tol_normals: stopping criterion on median angular difference between two normal maps
%			(default: 0.1)
%		- PARAMS.ratio: integer downsampling factor 
%			(default: 1)
%
%		- PARAMS.precond: preconditioner used
%		precond = 'cmg' uses Koutis conjugate multigrid preconditioner
%		precond = 'ichol' uses incomplete Cholesky 
%		precond = 'gs' uses Gauss-Seidel 
%		precond = 'jacobi' uses Jacobi 
%			(default: 'jacobi', but 'cmg' is strongly recommended)
%		- PARAMS.tol_pcg: relative stopping criterion for inner CG iter
%			(default: 1e-6)
%		- PARAMS.maxit_pcg: max iterations for inner CG iter
%			(default: 25)
%
%		- PARAMS.display: binary variable indicating if current shape
%		and albedo are displayed at each iteration 
%			(default: 0)
%
%
%	=== CREDITS ===
%
%	This is inspired by the robust PS method in :
%	[1]	"A Non-Convex Variational Approach to Photometric Stereo under Inaccurate Lighting"
%		Quéau et al.
%		Proc. CVPR 2017, Honolulu, USA 

	disp(' ');
	
	if(nargin<2)
		disp('ERROR: must provide data and calibration');
		return;
	end

	%% %%% Check DATA
	%%% Check DATA
	% Check images
	if(~isfield(data,'I'))
		disp('ERROR: images not provided in DATA.I')
		return;
	end
	I = double(data.I); clear data.I;
	if(size(I,4)==1) % Graylevel images
		nchannels = 1;
		[nrows,ncols,nimgs] = size(I);
	else % Color images
		[nrows,ncols,nchannels,nimgs] = size(I);
	end
	% Check mask
	if(~isfield(data,'mask'))
		disp('WARNING: mask not provided in DATA.mask, using full 2D grid')
		data.mask = ones(nrows,ncols);
	end
	mask = double(data.mask)>0; clear data.mask;
	if(size(mask,1)~=nrows | size(mask,2)~=ncols | ndims(mask)>2)
		disp(sprintf('ERROR: mask should be %d x %d',nrows,ncols));
		return;
	end
	% Check ambient
	if(~isfield(data,'ambient'))
		disp('WARNING: ambient not provided in DATA.ambient, assuming null')
		data.ambient = zeros(nrows,ncols,nchannels);
	end
	ambient = double(data.ambient); clear data.ambient; 
	clear data

	%% %%% Check CALIB
	%%% Check CALIB	
	% Check locations
	if(~isfield(calib,'S'))
		disp('ERROR: sources locations not provided in DATA.S')
		return;
	end
	S = double(calib.S); clear calib.S;
	if(size(S,3)~=nchannels)
		S = repmat(S,[1 1 nchannels]);
	end
	if(size(S,1)~=nimgs | size(S,2)~= 3 | size(S,3)~= nchannels) 
		disp(sprintf('ERROR: S should be %d x 3 x %d',nimgs,nchannels));
		return;
	end
	norm_S = sqrt(sum(S.^2,2));
	S = bsxfun(@rdivide,S,norm_S);
	% Check intrinsics
	if(~isfield(calib,'K')|size(calib.K,1)==0)
		disp('Warning: intrinsics not provided in CALIB.K - using orthographic projection')
		calib.K = [1,0,0.5*size(I,2);0,1,0.5*size(I,1);0,0,1];
		orthographic = 1;
	else
		orthographic = 0;
	end
	K = double(calib.K); clear calib.K;
	if(size(K,1)~=3 | size(K,2)~= 3 | ndims(K)>2) 
		disp('ERROR: K should be 3 x 3');
		return;
	end
	if(K(1,3)==0)
		K = K';
	end
	if(K(1,1)==0 | K(2,2)==0 | K(1,3)==0 | K(2,3)==0 | K(3,3)~=1 )
		disp('ERROR: intrinsics matrix not in a standard format');
		return;
	end
	% Check intensities
	if(~isfield(calib,'Phi'))
		disp('WARNING: intensities not provided in CALIB.Phi, using default values')
		calib.Phi = squeeze(norm_S);
	else 
		calib.Phi = calib.Phi.*squeeze(norm_S); 
	end
	Phi = double(calib.Phi); clear calib.Phi;
	if(size(Phi,1)~=nimgs | size(Phi,2)~=nchannels | ndims(Phi)>2)
		disp(sprintf('ERROR: Phi should be %d x %d',nimgs,nchannels));
		return;
	end
	clear calib

	%% %%% Check PARAMS
	%%% Check PARAMS
	if(~exist('params','var')|isempty(params)) params=[]; end;
	% Check initial depth
	if(~isfield(params,'z0'))
		disp('WARNING: initial depth not provided in PARAMS.z0, using default values')
		params.z0 = 700*ones(nrows,ncols);
	end
	z0 = params.z0; clear params.z0;
	if(size(z0,1)~=nrows | size(z0,2) ~= ncols | ndims(z0) > 2)
		disp(sprintf('ERROR: z0 should be %d x %d',nrows,ncols));
    end	
    % Check initial albedo
	if(~isfield(params,'initial_albedo'))
		disp('WARNING: initial albedo not provided in PARAMS.initial_albedo, using default values')
		params.initial_albedo = zeros(nrows,ncols,nchannels);
	end
	initial_albedo = params.initial_albedo; clear params.initial_albedo;
    % Check initial shininess
	if(~isfield(params,'initial_shininess'))
		disp('WARNING: initial shininess not provided in PARAMS.initial_shininess, using default values')
		params.initial_shininess = 1*ones(nrows,ncols,1);   % No shininess by default, to steer residuals on the left
	end
	initial_shininess = params.initial_shininess; clear params.initial_shininess;
    % Check initial rho_S
	if(~isfield(params,'initial_rho_S'))
		disp('WARNING: initial rho_S not provided in PARAMS.initial_rho_S, using default values')
		params.initial_rho_S = 1 * ones(nrows,ncols,1);   % .5 specular map, to make sure we start on the left of the estimator function
	end
	initial_rho_S = params.initial_rho_S; clear params.initial_rho_S;
	% Check prior
	if(~isfield(params,'prior'))
		disp('WARNING: prior not provided in PARAMS.prior, using default values')
		params.prior = 1e-6;
	end
	prior = params.prior; clear params.prior;
	if(prior<0)
		disp(sprintf('ERROR: prior should be positive'));
	end
	% Check stabilizer
	if(~isfield(params,'stabilizer'))
		disp('WARNING: stabilizer not provided in PARAMS.stabilizer, using default values')
		params.stabilizer = 0.1;
	end
	stabilizer = params.stabilizer; clear params.stabilizer;
	if(stabilizer<0)
		disp(sprintf('ERROR: stabilizer should be positive'));
	end	
	
	% Check scales
	if(~isfield(params,'scales'))
		disp('WARNING: scales not provided in PARAMS.scales, using default values')
		params.scales = 1;
	end
	scales = params.scales; clear params.scales;
	if(scales<0)
		disp(sprintf('ERROR: scales should be positive'));
	end			
	
	% Check semi_calibrated
	if(~isfield(params,'semi_calibrated'))
		disp('WARNING: semi_calibrated parameter not provided in PARAMS.semi_calibrated, using default values')
		params.semi_calibrated = 0;
	end
	semi_calibrated = params.semi_calibrated; clear params.semi_calibrated;
	if(semi_calibrated ~= 1 & semi_calibrated ~= 0)
		disp('ERROR: semi_calibrated should be binary');
	end
	% Check uncalibrated
	if(~isfield(params,'uncalibrated'))
		disp('WARNING: uncalibrated parameter not provided in PARAMS.uncalibrated, using default values')
		params.uncalibrated = 0;
	end
	uncalibrated = params.uncalibrated; clear params.uncalibrated;
	if(uncalibrated ~= 1 & uncalibrated ~= 0)
		disp('ERROR: uncalibrated should be binary');
	end	
	% Check ambient
	if(~isfield(params,'do_ambient'))
		disp('WARNING: do_ambient parameter not provided in PARAMS.do_ambient, using default values')
		params.do_ambient = 0;
	end
	do_ambient = params.do_ambient; clear params.do_ambient;
	if(do_ambient ~= 1 & do_ambient ~= 0)
		disp('ERROR: do_ambient should be binary');
		return;
    end	
    % Check albedo
	if(~isfield(params,'do_albedo'))
		disp('WARNING: do_albedo parameter not provided in PARAMS.do_albedo, using default values')
		params.do_albedo = 1;
	end
	do_albedo = params.do_albedo; clear params.do_albedo;
	if(do_albedo ~= 1 & do_albedo ~= 0)
		disp('ERROR: do_albedo should be binary');
		return;
    end	
    % Check depth
	if(~isfield(params,'do_depth'))
		disp('WARNING: do_depth parameter not provided in PARAMS.do_depth, using default values')
		params.do_depth = 1;
	end
	do_depth = params.do_depth; clear params.do_depth;
	if(do_depth ~= 1 & do_depth ~= 0)
		disp('ERROR: do_depth should be binary');
		return;
	end		
	
	% Check estimator
	if(~isfield(params,'estimator'))
		disp('WARNING: estimator not provided in PARAMS.estimator, using default values')
		params.estimator = 'LS';
	end
	estimator = params.estimator; clear params.estimator;
	% Check estimator's parameter
	if(~isfield(params,'lambda'))
		disp('WARNING: lambda parameter not provided in PARAMS.lambda, using default values')
		if(strcmp(estimator,'LS'))
			params.lambda = 0;
		elseif(strcmp(estimator,'Cauchy'))
			params.lambda = 0.15;
		elseif(strcmp(estimator,'Lp'))
			params.lambda = 0.7;
		elseif(strcmp(estimator,'GM'))
			params.lambda = 0.4;
		elseif(strcmp(estimator,'Welsh'))
			params.lambda = 0.4;
		elseif(strcmp(estimator,'Tukey'))
			params.lambda = 0.9;
		end
	end
	lambda = params.lambda; clear params.lambda;
    if(~isfield(params,'lambda_right'))
        params.lambda_right = lambda;
    end
    lambda_right = params.lambda_right; clear params.lambda_right;
    if(~isfield(params,'lambda_spec'))
        params.lambda_spec = 1;
    end
    lambda_spec = params.lambda_spec; clear params.lambda_spec;
	if(~strcmp(estimator,'LS') & lambda == 0)
		disp('ERROR: a strictly positive estimator parameter must be set in PARAMS.lambda for robust M-estimation');
	end
	% Check self shadows
	if(~isfield(params,'self_shadows'))
		disp('WARNING: self_shadows parameter not provided in PARAMS.self_shadows, using default values')
		params.self_shadows = 1;
	end
	self_shadows = params.self_shadows; clear params.self_shadows;
	if(self_shadows ~= 1 & self_shadows ~= 0)
		disp('ERROR: self_shadows should be binary');
		return;
	end
	% Check used
	if(~isfield(params,'used'))
		disp('WARNING: used not provided in PARAMS.used, using all data')
		params.used = ones(size(I));
	end
	used = double(params.used); clear params.used; 

	% Check max iterations
	if(~isfield(params,'maxit'))
		disp('WARNING: max number of iterations not set in PARAMS.maxit, using default values')
		params.maxit = 100;
	end
	maxit = params.maxit; clear params.maxit;
	if(maxit < 0)
		disp('ERROR: max number of iterations should be positive');
		return;
	end
	% Check tolerance
	if(~isfield(params,'tol'))
		disp('WARNING: tolerance not set in PARAMS.tol, using default values')
		params.tol = 1e-3;
	end
	tol = params.tol; clear params.tol;
	if(tol < 0)
		disp('ERROR: tolerance should be positive');
		return;
	end
	% Check tolerance normals
	if(~isfield(params,'tol_normals'))
		disp('WARNING: tolerance on normaps not set in PARAMS.tol_normals, using default values')
		params.tol_normals = 0.1;
	end
	tol_normals = params.tol_normals; clear params.tol_normals;
	if(tol_normals < 0)
		disp('ERROR: tol_normals should be positive');
		return;
	end	
	% Check ratio
	if(~isfield(params,'ratio'))
		disp('WARNING: downsampling factor not set in PARAMS.ratio, using default values')
		params.ratio = 1;
	end
	ratio = params.ratio; clear params.ratio;
	if(ratio < 0)
		disp('ERROR: ratio should be a positive integer');
		return;
	end
	if(ratio>1)
		K(1:2,:) = K(1:2,:)./ratio;
		mask = imresize(mask,1./ratio);
		z0 = imresize(z0,1./ratio);
		ambient = imresize(ambient,1./ratio);
		disp('rescaling images');
		if(size(I,4)>1)
			I_resized = imresize(I(:,:,:,1),1./ratio);
			used_resized = imresize(used(:,:,:,1),1./ratio);
			for im = 1:size(I,4)
				I_resized(:,:,:,im) = imresize(I(:,:,:,im),1./ratio); 
				used_resized(:,:,:,im) = imresize(used(:,:,:,im),1./ratio); 
			end
			I = I_resized;
			clear I_resized
			used = used_resized;
			clear used_resized
		else
			I_resized = imresize(I(:,:,1),1./ratio);
			used_resized = imresize(used(:,:,1),1./ratio);
			for im = 1:size(I,3)
				I_resized(:,:,im) = imresize(I(:,:,im),1./ratio); 
				used_resized(:,:,im) = imresize(used(:,:,im),1./ratio); 
			end
			I = I_resized;
			used = used_resized;
			clear I_resized
			clear used_resized
		end
		if(orthographic)
			K(1,1) = 1;
			K(2,2) = 1;
		end
		[nrows,ncols] = size(mask);
    end

    I_tot = I;  % we have I: diffuse image, I_tot: total image
	
	% Check preconditioner
	if(~isfield(params,'precond'))
		disp('WARNING: preconditioner not provided in PARAMS.precond, using default values')
		params.precond = 'ichol';
	end
	precond = params.precond; clear params.precond;	
	if(~strcmp(precond,'cmg') & ~strcmp(precond,'ichol') & ~strcmp(precond,'jacobi') & ~strcmp(precond,'gs'))
		disp('ERROR: unknown preconditioner');
		return;
	end
	% Check PCG tolerance
	if(~isfield(params,'tol_pcg'))
		disp('WARNING: PCG tolerance not set in PARAMS.tol_pcg, using default values')
		params.tol_pcg = 1e-6;
	end
	tol_pcg = params.tol_pcg; clear params.tol_pcg;
	if(tol_pcg < 0)
		disp('ERROR: PCG tolerance should be positive');
		return;
	end
	% Check PCG max iterations
	if(~isfield(params,'maxit_pcg'))
		disp('WARNING: max number of iterations for PCG not set in PARAMS.maxit_pcg, using default values')
		params.maxit_pcg = 25;
	end
	maxit_pcg = params.maxit_pcg; clear params.maxit_pcg;
	if(maxit_pcg < 0)
		disp('ERROR: max number of iterations for PCG should be positive');
		return;
    end
    
    % Check my additions
    if(~isfield(params,'correct_psi_chi'))
		disp('WARNING: did not specify whether to do the right psi, chi updates, using default values')
		params.correct_psi_chi = 1;
	end
	correct_psi_chi = params.correct_psi_chi; clear params.correct_psi_chi;
    if(~isfield(params,'correct_clamp'))
		disp('WARNING: did not specify whether to correct the excessive clamp for rho, using default values')
		params.correct_clamp = 0;
	end
	correct_clamp = params.correct_clamp; clear params.correct_clamp;
    if(~isfield(params,'do_specular'))
		disp('WARNING: did not specify whether to do specular optimization, using default values')
		params.do_specular = 0;
	end
	do_specular = params.do_specular; clear params.do_specular;
    if(~isfield(params,'spec_thresh'))
		disp('WARNING: did not specify when considering I_specular = 0, using default values')
		params.spec_thresh = 1e-12;
	end
	spec_thresh = params.spec_thresh; clear params.spec_thresh;
    if(~isfield(params,'scalar_shininess'))
		disp('WARNING: did not specify whether to estimate a map of shininesses, using default values')
		params.scalar_shininess = 1;
	end
	scalar_shininess = params.scalar_shininess; clear params.scalar_shininess;
    
	% Check display
	if(~isfield(params,'display'))
		disp('WARNING: display parameter not provided in PARAMS.display, using default values')
		params.display = 0;
	end
	display = params.display; clear params.display;
	if(display ~= 1 & display ~= 0)
		disp('ERROR: display should be binary');
		return;
	end
	if(display)
		hfig1 = figure();
		hfig2 = figure();
		hfig3 = figure();
		hfig4 = figure();
		hfig5 = figure();
		if(do_ambient)
			hfig6 = figure();
		end
	end
	
	clear params
	disp(' ');
	

	
	%% %%% Prepare data
	%%% Prepare data
	% Intrinsics
	fx = K(1,1);
	fy = K(2,2);
	x0 = K(1,3);
	y0 = K(2,3);
	clear K
	
	% Estimator phi(x) and weight function w(x) = phi'(x)/x
    % It should be down, so as to change it when scaling the lambdas, but
    % this produces bad results
	if(strcmp(estimator,'LS'))
		phi_fcn = @(x) 0.5*x.^2;	
		w_fcn = @(x) ones(size(x));
	elseif(strcmp(estimator,'Cauchy'))
        phi_fcn = @(x) 0.5*log(1+x.^2./(lambda.*(x<0)+lambda_right.*(x>=0)).^2);
%     	phi_fcn_right = @(x) 0.5*log(1+x.^2/lambda_right^2);
	    w_fcn = @(x) 1./((lambda.*(x<0)+lambda_right.*(x>=0)).^2+x.^2);
%         w_fcn_right = @(x) 1./(lambda_right^2+x.^2);
%         phi_fcn = @(x) phi_fcn_right(max(0,x))+ phi_fcn_left(min(0,x));
%         w_fcn = @(x) w_fcn_left(x).*(x<0) + w_fcn_right(x).*(x>=0);
%         phi_fcn = @(x) 0.5*log(1+x.^2/lambda^2);	
% 	    w_fcn = @(x) 1./(lambda^2+x.^2);
        phi_fcn_spec = @(r, s) log(1+(s/lambda_spec).^2.*(1-exp(-r)).^2);
        w_fcn_spec = @(r, s) (2*s.^2.*(exp(r)-1))./(r.*(lambda_spec^2*exp(2*r)+s.^2.*(exp(r)-1).^2));
	elseif(strcmp(estimator,'Lp'))
		thr_norm = 1e-2;
		phi_fcn = @(x) ((abs(x)).^lambda) ./ (lambda*(thr_norm^(lambda-2)));
		w_fcn = @(x) lambda.*(max(thr_norm,abs(x))).^(lambda-2) ./ (lambda*(thr_norm^(lambda-2))) ;  
	elseif(strcmp(estimator,'GM'))
		phi_fcn = @(x) 0.5*(lambda^2)*x.^2./(x.^2+lambda^2);
		w_fcn = @(x) lambda^4./((x.^2+lambda^2).^2);	
	elseif(strcmp(estimator,'Welsh'))
		phi_fcn = @(x) 0.5*lambda^2.*(1-exp(-x.^2./(lambda^2)));
		w_fcn = @(x) exp(-x.^2./(lambda^2));
	elseif(strcmp(estimator,'Tukey'))
		phi_fcn = @(x) (1/6)*(abs(x)<=lambda).*(1-(1-x.^2./(lambda^2)).^3).*lambda^2+(abs(x)<lambda).*lambda^2;
		w_fcn = @(x) (abs(x)<=lambda).*((1-x.^2./(lambda^2)).^2);
	end
	% Self shadow function psi(x) and derivative psi'(x)
	if(self_shadows)
		psi_fcn = @(x) max(x,0);     
		chi_fcn = @(x) double(x>=0); 
	else
		psi_fcn = @(x) x;   
		chi_fcn = @(x) 1*ones(size(x));	
	end
	max_I = max(I(:));  % obs: I = rho * max(0, n@l) + amb (probably) -> I can do this normalization! But be sure to un-normalize
    I = I./max_I;
    I_tot = I_tot./max_I;   % I just the diffuse + ambient image?
	ambient = ambient./max_I;
	Phi = Phi./max_I;   % For specular: I = rho*max(0, n@s) phi + amb + rho_S (...)^gamma*max(0, n@s) phi -> phi takes the rescaling
	
	%% %%% Initial and final values
	% Store the initial intrinsics
	x0_ref = x0;
	y0_ref = y0;
	fx_ref = fx;
	fy_ref = fy;
	% Store the initial images and masks
	I_ref = I;
    I_tot_ref = I_tot;
	mask_ref = mask;
	used_ref = used; 
	nrows_ref = size(I_ref,1);
	ncols_ref = size(I_ref,2);
	lambda_ref = lambda;
    lambda_ref_right = lambda_right;
	% Store initials priors
	z0_ref = z0;
	% Store output
	z_final = z0;
    rho_final = imresize(initial_albedo, 1./ratio);
	ambient_final = ambient; 
    shininess_final = imresize(initial_shininess, 1./ratio);
    rho_S_final = imresize(initial_rho_S, 1./ratio);
	% Set the scales
	scales = 2.^(scales-1:-1:0);    % default is 1, i.e. do not scale (keep original)
    % Set mu_cos, lambda_cos
    mu_cos = 1; lambda_cos = 1.93;
	%% Main stuff
	
    disp("Have I screwed something up in the updates? Are all optimization variables independent?")
    pause;
	clear ratio
	for ratio=scales

        start_specular_optimization = 1;

		% Scale inputs
		if(~orthographic)
			fx = fx_ref./ratio;
			fy = fy_ref./ratio;
		end
		x0 = x0_ref./ratio;
		y0 = y0_ref./ratio;
		mask = imresize(mask_ref,1./ratio)>0;
		z0 = imresize(z0_ref,1./ratio); % could have been z_final, as z_ref is only used here
% 		ambient = imresize(ambient_final,1./ratio); % Error, already done below
		disp('rescaling images');
		if(size(I_ref,4)>1)
			I_resized = imresize(I_ref(:,:,:,1),1./ratio);
            I_tot_resized = imresize(I_tot_ref(:,:,:,1),1./ratio);
			used_resized = imresize(used_ref(:,:,:,1),1./ratio)>0;
			for im = 1:size(I_ref,4)
				I_resized(:,:,:,im) = imresize(I_ref(:,:,:,im),1./ratio); 
                I_tot_resized(:,:,:,im) = imresize(I_tot_ref(:,:,:,im),1./ratio); 
				used_resized(:,:,:,im) = imresize(used_ref(:,:,:,im),1./ratio); 
			end
			I = I_resized;
            I_tot = I_tot_resized;
			clear I_resized I_tot_resized
			used = used_resized;
			clear used_resized
		else
			I_resized = imresize(I_ref(:,:,1),1./ratio);
            I_tot_resized = imresize(I_tot_ref(:,:,1),1./ratio);
			used_resized = imresize(used_ref(:,:,1),1./ratio)>0;
			for im = 1:size(I_ref,3)
				I_resized(:,:,im) = imresize(I_ref(:,:,im),1./ratio); 
                I_tot_resized(:,:,im) = imresize(I_tot_ref(:,:,im),1./ratio); 
				used_resized(:,:,im) = imresize(used_ref(:,:,im),1./ratio); 
			end
			I = I_resized;
            I_tot = I_tot_resized;
			clear I_resized I_tot_resized
			used = used_resized;
			clear used_resized
		end
		[nrows,ncols] = size(mask);
		[uu,vv] = meshgrid(1:ncols,1:nrows);
		u_tilde = (uu - x0); 
		v_tilde = (vv - y0);
		clear uu vv x0 y0
		% Mask
		imask = find(mask>0);
		npix = length(imask);
		% Some useful change
		if(nchannels>1)
			I = permute(I,[1 2 4 3]);   % now n row, ncols, nimgs, nchannels
            I_tot = permute(I_tot,[1 2 4 3]);   % now n row, ncols, nimgs, nchannels
		end
		% Some useful variables
		px_rep = repmat(u_tilde(imask),[1 nimgs]);
		py_rep = repmat(v_tilde(imask),[1 nimgs]);
		if(orthographic)
			px_rep(:) = 0;
			py_rep(:) = 0;
		end
		% Vectorize images
		I = reshape(I,nrows*ncols,nimgs,nchannels);
		I = I(imask,:,:);
        I_tot = reshape(I_tot,nrows*ncols,nimgs,nchannels);
		I_tot = I_tot(imask,:,:);
		% Vectorize binary mask of used data
		W_idx = reshape(used,nrows*ncols,nimgs,nchannels);
		W_idx = W_idx(imask,:,:);
		clear used
		
		% Scale parameters
		if(~strcmp(estimator,'Lp'))
			lambda = lambda_ref*median(abs(I(:)-median(I(:))));
            lambda_right = lambda_ref_right*median(abs(I(:)-median(I(:))));
		end
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%% Initialize variables at current scale
		z = imresize(z_final,1./ratio);
		z(mask==0) = NaN;
		rho = imresize(rho_final,1./ratio);
		ambient = imresize(ambient_final,1./ratio);
        rho_S = imresize(rho_S_final,1./ratio);
        shininess = imresize(shininess_final, 1./ratio);

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%% Auxiliary variables at current scale		
		% Gradient operator on the mask
		G = make_gradient(mask);
		Dx =  G(1:2:end,:); % size # valid pixels x # valid pixels
		Dy = G(2:2:end,:);  % size # valid pixels x # valid pixels
		
		clear G;
		if(orthographic)
			z_tilde = z(imask);
			z0 = z0(imask);
			XYZ = cat(3,u_tilde./fx,v_tilde./fy,z);
		else
			z0 = log(z0(imask));
			z_tilde = log(z(imask));
			XYZ = cat(3,z.*u_tilde./fx,z.*v_tilde./fy,z);   % XYZ(u,v,:) is the 3D point in camera space, corresponding to the pixel (u,v)
		end
		zx = Dx*z_tilde;
		zy = Dy*z_tilde;
		Nx = zeros(nrows,ncols);
		Ny = zeros(nrows,ncols);
		Nz = zeros(nrows,ncols);
		Nx(imask) = fx*zx;
		Ny(imask) = fy*zy;
		if(orthographic)
			Nz(imask) = -1;
		else
			Nz(imask) = -u_tilde(imask).*zx-v_tilde(imask).*zy-1;
		end
		dz = sqrt(Nx.^2+Ny.^2+Nz.^2);
		N = cat(3,Nx./dz,Ny./dz,Nz./dz);    % the normals: N(u,v,:) = normal at (u,v)
		rho_tilde = reshape(bsxfun(@rdivide,rho,dz),nrows*ncols,nchannels);
		rho_tilde = rho_tilde(imask,:);
		ambient_tilde = reshape(ambient,nrows*ncols,nchannels);
		ambient_tilde = ambient_tilde(imask,:);
		tab_nrj = [];
		
		disp('==============================================================');
		disp(sprintf('Starting algorithm at scale %d with %d pixels, %d images, %d channels, lambda = %.6f',ratio,npix,nimgs,nchannels,lambda));
		disp(' ')
	
	
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%% Initial energy
		
		shading_fcn = @(z,S) (spdiags(reshape(bsxfun(@minus,fx*transpose(S(:,1)),bsxfun(@times,px_rep,transpose(S(:,3)))),npix*nimgs,1),0,npix*nimgs,npix*nimgs)*repmat(Dx,[nimgs 1])+spdiags(reshape(bsxfun(@minus,fy*transpose(S(:,2)),bsxfun(@times,py_rep,transpose(S(:,3)))),npix*nimgs,1),0,npix*nimgs,npix*nimgs)*repmat(Dy,[nimgs 1]))*z-reshape(transpose(repmat(S(:,3),[1 npix])),npix*nimgs,1);
		r_fcn = @(rho,shadz,II,phi,amb,W_idx) W_idx.*(reshape(rho*phi',[nimgs*npix 1]).*psi_fcn(shadz)+repmat(amb,[nimgs 1])-II); % residual
		J_fcn = @(rho,shadz,II,phi,amb,W_idx) sum(phi_fcn(r_fcn(rho,shadz,II,phi,amb,W_idx))); % energy
		energy = 0;
		psi = zeros(npix,nimgs,nchannels);
		for ch = 1:nchannels
			psi(:,:,ch) = reshape(shading_fcn(z_tilde,S(:,:,ch)),npix,nimgs);   % this is actually the argument of psi_fcn
			psich = psi(:,:,ch);
			phich = Phi(:,ch);
			Ich = I(:,:,ch);
			Wch = W_idx(:,:,ch);
			energy = energy+J_fcn(rho_tilde(:,ch),psich(:),Ich(:),phich,ambient_tilde(:,ch),Wch(:));
		end
		energy = energy./(npix*nimgs*nchannels);    % normalize by # valid pixels x # channels x # images
		clear Ich psich
		disp(sprintf('== it. 0 - energy : %.20f',energy));
		disp(' ');
		tab_nrj(1) = energy;
		

		% Display Initial result
		if(display)
			figure(hfig1);
			surfl(-XYZ(:,:,1),-XYZ(:,:,2),-XYZ(:,:,3),[0 90]);
			shading flat;
			colormap gray
			view(170,30);
			axis ij
			axis image
			title('Shape')

			figure(hfig2)
			semilogy(0:0,tab_nrj(1:1),'Linewidth',4)
			title('Energy')


			figure(hfig3)
			imagesc(rho./max(rho(:)))
			if(nchannels==1)
				colormap gray
				colorbar
			end
			axis image
			axis off
			title('Albedo')

			figure(hfig4)
			Ndisp = N;
			Ndisp(:,:,3) = -Ndisp(:,:,3);
			Ndisp = 0.5*(Ndisp+1);
			imagesc(Ndisp)
			axis image
			axis off
			title('Normals')
			drawnow
			
			if(do_ambient)
				figure(hfig6)
				amb_disp = ambient./max(ambient(:));
				imagesc(uint8(255*amb_disp))
				if(nchannels==1)
					colormap gray
				end
				axis image
				axis off
				title('Ambient')
			end
		end
		
		for it = 1:maxit %%%%%%%%%%%%%%%%%%%%%%%% OPTIMIZATION STARTS HERE!
			w = zeros(npix,nimgs,nchannels);
			chi = chi_fcn(psi);

			% Pseudo-albedo update
			for ch = 1:nchannels
				Ich = I(:,:,ch);
				Wch = W_idx(:,:,ch);
				psich = psi(:,:,ch);
				phich = Phi(:,ch);
				phi_chich = bsxfun(@times,psi(:,:,ch).*chi(:,:,ch),transpose(Phi(:,ch)));
				w(:,:,ch) = Wch.*w_fcn(reshape(r_fcn(rho_tilde(:,ch),psich(:),Ich(:),phich,ambient_tilde(:,ch),Wch(:)),npix,nimgs));
				wch = w(:,:,ch);
				if(do_ambient)  % don't worry for the moment about what happens with do_ambient = 1
					if(do_albedo)
						A11 = sum(w(:,:,ch).*(phi_chich).^2,2);
						A12 = sum(w(:,:,ch).*(phi_chich),2);
						A22 = sum(w(:,:,ch),2);
						b1 = sum(w(:,:,ch).*I(:,:,ch).*phi_chich(:,:),2);
						b2 = sum(w(:,:,ch).*I(:,:,ch),2);
						det_A = A11.*A22-A12.^2;
						idx_ok = find(det_A>0);
						rho_tilde(idx_ok,ch) = (A22(idx_ok).*b1(idx_ok)-A12(idx_ok).*b2(idx_ok))./det_A(idx_ok);
						ambient_tilde(idx_ok,ch) = (-A12(idx_ok).*b1(idx_ok)+A11(idx_ok).*b2(idx_ok))./det_A(idx_ok);
					else
						denom = (sum(w(:,:,ch),2));
						idx_ok = find(denom>0);
						if(length(idx_ok>0))
							ambient_tilde(idx_ok,ch) = (sum(w(idx_ok,:,ch).*(I(idx_ok,:,ch)-bsxfun(@times,phi_chich(idx_ok,:),rho_tilde(idx_ok,ch))),2))./denom(idx_ok);
						end
					end				
				else
					if(do_albedo)
						denom = (sum(w(:,:,ch).*(phi_chich).^2,2));
						idx_ok = find(denom>0);
						if(length(idx_ok>0))
							rho_tilde(idx_ok,ch) = (sum(w(idx_ok,:,ch).*(bsxfun(@minus,I(idx_ok,:,ch),ambient_tilde(idx_ok,ch))).*phi_chich(idx_ok,:),2))./denom(idx_ok);
						end
					else
						rho_ch = rho(:,:,ch);
						rho_tilde(:,ch) = rho_ch(imask)./dz(imask);
					end
				end
			end
			rho_tilde = max(0,rho_tilde);
			clear psich phi_chich Ich rho_ch

            % Here I should be upper clamp rho!
            % Update rho, rho_tilde
			for ch = 1:nchannels
				rhoch = rho(:,:,ch);
				rhoch(imask) = rho_tilde(:,ch).*dz(imask);
				Ich = I(:,:,ch);
				Phich = Phi(:,ch);
                psich = psi(:,:,ch);
                if correct_clamp
                    rho(:,:,ch) = min(max(Ich(:))./(min(psich(:)).*min(Phich(:))),max(0,rhoch)); % [ADDITION] max(0, n@S) was not taken into account!				
                else
                    rho(:,:,ch) = min(max(Ich(:))./min(Phich(:)),max(0,rhoch)); % WRONG, but performs better
                end

                %   GET THE CORRECT CLAMPED VERSION OF RHO_TILDE HERE!
                rho_tilde_tmp_ch =  rho(:,:,ch)./dz;
                rho_tilde(:,ch) = rho_tilde_tmp_ch(imask);
            end

            
            % Specular stuff (INEFFICIENT)
            I_S_full = [];
            if do_specular>0 && start_specular_optimization==1   % create tilde variables also for the specular case. Take care to phi! Also, beware of the different modes
                if do_specular == 1
                    % It means, we only optimize over shininess and rho_S

                    % Compute log(I_S)
                    % Compute I_S... the residual between I and lambert +
                    % ambient stuff
                    I_S = zeros(size(I));   % # valid pixels x # images x # channels
                    for ch = 1:nchannels
				        Ich = I_tot(:,:,ch);
				        psich = reshape(shading_fcn(z_tilde,S(:,:,ch)),npix,nimgs);     % recomputing it for good measure
                        residual_spec_ch = -r_fcn(rho_tilde(:,ch),psich(:),Ich(:),Phi(:,ch),ambient_tilde(:,ch),1);
				        I_S(:,:,ch) = reshape(residual_spec_ch, size(Ich));   % image - stuff, no residual weighting
                    end
                    clear residual_spec_ch psich Ich;

                    % Compute h . n
                    view_vectors = (0-XYZ)./vecnorm(XYZ,2,3);   % 2 norm along direction 3
                    light_directions = permute(S,[1,3,2]);           % it is now n_frame x nchannels x 3
                    h = tensor_sum(view_vectors, light_directions);
                    h = h./vecnorm(h,2,length(size(h)));    % so, we index h as follows: (u,v,im,ch,:)
                    N_exp = permute(N, [1,2,4,5,3]);
                    N_exp = repmat(N_exp,[1,1,nimgs,nchannels,1]);
                    hn =sum(h.*N_exp,5);    % a variable u, v, im, ch 

                    % Compute S . n
                    sn = tensorprod(light_directions, N, 3);    % https://www.mathworks.com/matlabcentral/answers/1690105-dot-product-of-tensor
                    
                    alpha_tilde = hn - 1;
                    beta_tilde = permute(lambda_cos*(sn-1)+real(log(Phi)),[3,4,1,2]);    % could have absorbed Phi into eta_tilde... and no need for reshaping shenanigans, and in coherence with alpha_tilde

                    % Now, I_S_ch might really be < 0 as albedos or ambient
                    % might be wrong! To prove so, I store the specular
                    % images in full and then output it
                    rmask = repmat(mask,[1,1, size(I_S, 2), size(I_S, 3)]);
                    I_S_full = NaN * ones(size(rmask));
                    I_S_full(rmask) = I_S(:);
                    clear rmask;
                    I_S_full = permute(I_S_full, [1,2,4,3]);

                    I_tilde_no_log = permute(I_S_full, [1,2,4,3]);
                    I_tilde = real(log(I_tilde_no_log)); 

                    sn = permute(sn, [3,4,1,2]);
                    
                    E = []; % Errors
                    for k=1:20                      
                            
                        % Refresh the mask
                        rmask = repmat(mask,[1,1, size(I_S, 2), size(I_S, 3)]);
                        rmask = logical( rmask .* (permute(I_S_full, [1,2,4,3]) > spec_thresh));
                        
                        % Log the energy
                        I_S_est = exp(shininess.*(hn-1)+lambda_cos*(sn-1)).*permute(repmat(Phi,[1,1,size(hn,1), size(hn,2)]),[3,4,1,2]) .* mu_cos .* rho_S.* (shininess+2);
                        I_tilde_no_log(~rmask) = 0;
                        I_S_est(~rmask) = 0;
                        e = norm((I_S_est-I_tilde_no_log).*rmask, 'fro')/numel(rmask(rmask>0));
                        E = [E,e];              

                        % Compute the interesting variables
                        eta_tilde = (shininess+2).*rho_S.*mu_cos;  % (u,v), full storage
                        eta_tilde = repmat(eta_tilde, [1,1,nimgs, nchannels]);   % also (u,v,nimgs,nchannels)
                        rmask = logical( rmask .* (eta_tilde > spec_thresh));
                        eta_tilde = real(log(eta_tilde));
                        
                        % Removing, in theory, lights are never zero...
%                         Phi_mask = permute(repmat(Phi > spec_thresh,[1,1,size(eta_tilde,1), size(eta_tilde,2)]), [3, 4, 1, 2]);
%                         rmask = logical( rmask .* Phi_mask);
                        
                        shininess_tilde = repmat(shininess, [1,1,nimgs, nchannels]);
    
                        % Where there are NaNs or logarithm of negative values, just put 0
                        shininess_tilde(~rmask) = 0;
                        eta_tilde(~rmask) = 0;
                        alpha_tilde(~rmask) = 0;
                        beta_tilde(~rmask) = 0;
                        I_tilde(~rmask) = 0;
                        I_tilde_no_log(~rmask) = 0;
    
                        % Updating shininess
                        % Residuals, weights
                        res_spec = (I_tilde - beta_tilde) - ...
                            (eta_tilde + shininess_tilde .* alpha_tilde);
                        w_shininess_tilde = real(w_fcn_spec(res_spec, I_tilde_no_log));
                        w_shininess_tilde(~rmask) = 0;
                        % Least squares with handcrafted estimator
                        w_shininess_tilde = reshape(w_shininess_tilde,size(w_shininess_tilde,1), size(w_shininess_tilde,2), []);
                        alpha_tilde = reshape(alpha_tilde, size(alpha_tilde,1), size(alpha_tilde,2), []);   % back to a pixel map
                        denom_shininess_tilde = sum(w_shininess_tilde.*alpha_tilde.^2, n_dim(w_shininess_tilde));
                        if scalar_shininess == 1
                            denom_shininess_tilde = sum(denom_shininess_tilde, 'all');
                        end
                        num_shininess_tilde = (I_tilde - beta_tilde) - eta_tilde;
                        num_shininess_tilde = reshape(num_shininess_tilde, size(num_shininess_tilde,1), size(num_shininess_tilde,2),[]);
                        num_shininess_tilde = sum(num_shininess_tilde.*w_shininess_tilde.*alpha_tilde, n_dim(num_shininess_tilde));
                        if scalar_shininess == 1
                            num_shininess_tilde = sum(num_shininess_tilde,'all');
                        end
                        shininess_update_mask = denom_shininess_tilde>0;
                        upd_shininess_tilde = num_shininess_tilde(shininess_update_mask)./denom_shininess_tilde(shininess_update_mask);
                        shininess_tilde = shininess_tilde(:,:,1,1);
                        if scalar_shininess == 1
                            shininess_tilde = 0;   % who cares, I will 99% update it in a line...   IMPROVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        end
                        shininess_tilde(shininess_update_mask) = upd_shininess_tilde;
                        % Fix back the variables
                        if scalar_shininess == 0
                            shininess_tilde = max(repmat(shininess_tilde, [1, 1, nimgs, nchannels]),0);
                        else
                            shininess_tilde = max(shininess_tilde * ones(nrows, ncols, nimgs, nchannels),0);
                        end
                        alpha_tilde = reshape(alpha_tilde, nrows, ncols, nimgs, nchannels);
    
                        % Updating eta_tilde
                        % Residuals, weights
                        res_spec = (I_tilde - beta_tilde) - ...
                            (eta_tilde + shininess_tilde .* alpha_tilde);
                        w_eta_tilde = w_fcn_spec(res_spec, I_tilde_no_log);
                        w_eta_tilde(~rmask) = 0;
                        % Least squares with handcrafted estimator
                        w_eta_tilde = reshape(w_eta_tilde,size(w_eta_tilde,1), size(w_eta_tilde,2), []);    % reshape into (u,v ...)
                        denom_eta_tilde = sum(w_eta_tilde, n_dim(w_eta_tilde)); % note: for (u,v), if a single weight is valid, we'll consdider him
                        num_eta_tilde = (I_tilde - beta_tilde) - (shininess_tilde .* alpha_tilde);
                        num_eta_tilde  =  reshape(num_eta_tilde,size(num_eta_tilde,1), size(num_eta_tilde,2), []);
                        num_eta_tilde = sum(num_eta_tilde.*w_eta_tilde, n_dim(num_eta_tilde));
                        eta_update_mask = denom_eta_tilde>0;
                        upd_eta_tilde = num_eta_tilde(eta_update_mask)./denom_eta_tilde(eta_update_mask);
                        eta_tilde = eta_tilde(:,:,1,1);     % back to a pixel map
                        eta_tilde(eta_update_mask) = upd_eta_tilde;
    
                        % Recover original variables
                        shininess = shininess_tilde(:,:,1,1);
                        rho_S = max(exp(eta_tilde)./((shininess+2)*mu_cos), 0);

                    end

%                     plot(log(E));

                    % Subtract from total
                    % image! And be careful can I make the total image
                    % negative? Yes...
                    specular_update_mask = logical((1-scalar_shininess)*shininess_update_mask+eta_update_mask);
                    imask_specular = find(specular_update_mask>0);
                    [~, I_diffuse_imask, ~] = intersect(imask, imask_specular);
                    I_S_est = exp(shininess.*(hn-1)+lambda_cos*(sn-1)).*permute(repmat(Phi,[1,1,size(hn,1), size(hn,2)]),[3,4,1,2]) .* mu_cos .* rho_S.* (shininess+2);
                    I_S_est = reshape(I_S_est, nrows*ncols, nimgs, nchannels);
                    I(I_diffuse_imask,:,:) = max(I_tot(I_diffuse_imask,:,:) - I_S_est(imask_specular,:,:),0);

                    close all; i = 1;
                    subplot(1,4,1); imshow(max_I*imresize(squeeze(I_ref(:,:,:,i)), 1./ratio));
                    subplot(1,4,2); imshow(max_I*squeeze(I_S_full(:,:,:,i)));
                    subplot(1,4,3); imshow(max_I*(imresize(squeeze(I_ref(:,:,:,i)), 1./ratio)-max(reshape(squeeze(I_S_est(:,i,:)),[nrows, ncols, nchannels]),0)));
                    subplot(1,4,4); imshow(max_I*reshape(squeeze(I_S_est(:,i,:)),[nrows, ncols, nchannels]));

                    median(shininess_tilde(mask))
                    median(rho_S(mask))
    
                    clear shininess_update_mask eta_update_mask specular_update_mask;
                    clear w_eta_tilde num_eta_tilde denom_eta_tilde upd_eta_tilde;
                    clear w_shininess_tilde denom_shininess_tilde num_shininess_tilde upd_eta_tilde;
                    clear view_vectors light_directions N_exp h hn sn...
                        psich Ich rmask I_tilde eta_tilde alpha_tilde...
                        beta_tilde res_spec w_spec shininess_tilde   ...
                        Phi_mask I_tilde_no_log;

                else
                    disp("This specular mode is not yet implemented");
                    return;
                end
            end


			% Log-depth update
			if(do_depth)
				rho_rep = zeros(npix,nimgs*nchannels);
				for ch = 1:nchannels
					Ich = I(:,:,ch);
					Wch = W_idx(:,:,ch);
					psich = psi(:,:,ch);
					phich = Phi(:,ch);
					w(:,:,ch) = Wch.*w_fcn(reshape(r_fcn(rho_tilde(:,ch),psich(:),Ich(:),phich,ambient_tilde(:,ch),Wch(:)),npix,nimgs));
					rho_rep(:,(ch-1)*nimgs+1:ch*nimgs) = rho_tilde(:,ch)*transpose(phich);
				end
				
				D = reshape(chi,npix,nimgs*nchannels).*(rho_rep.^2).*reshape(w,npix,nimgs*nchannels);
				clear rho_rep

				A = [];
				rhs = [];
				for ch = 1:nchannels
					Ich = I(:,:,ch);
					A =[A;(spdiags(reshape(bsxfun(@minus,fx*transpose(S(:,1,ch)),bsxfun(@times,px_rep,transpose(S(:,3,ch)))),npix*nimgs,1),0,npix*nimgs,npix*nimgs)*repmat(Dx,[nimgs 1])+spdiags(reshape(bsxfun(@minus,fy*transpose(S(:,2,ch)),bsxfun(@times,py_rep,transpose(S(:,3,ch)))),npix*nimgs,1),0,npix*nimgs,npix*nimgs)*repmat(Dy,[nimgs 1]))];
					rho_rep_ch = rho_tilde(:,ch)*transpose(Phi(:,ch));
					rhs_ch = chi(:,:,ch).*rho_rep_ch.*(bsxfun(@times,rho_rep_ch,transpose(S(:,3,ch)))+bsxfun(@minus,Ich,ambient_tilde(:,ch))).*w(:,:,ch);
					rhs = [rhs;rhs_ch(:)];
				end
				At = transpose(A);
				M = At*spdiags(D(:),0,nimgs*npix*nchannels,nimgs*npix*nchannels)*A;
					
				M = M/(nimgs*npix*nchannels)+prior*speye(npix)/npix;
					
				rhs = At*rhs(:)/(nimgs*npix*nchannels)+prior*z0/npix;
				clear A At rho_rep_ch rhs_ch
				
				% Recompute the preconditioner every 5 iterations
				if(mod(it,5)==1)
					disp('Preconditioner recomputed')
					disp(' ');
					if(strcmp(precond,'cmg'))
						precond_L = cmg_sdd(M);
						precond_R = [];
					elseif(strcmp(precond,'ichol'))
						precond_L = ichol(M);
						precond_R = precond_L';
					elseif(strcmp(precond,'gs'))
						D = spdiags(spdiags(M,0),0,size(M,1),size(M,2));
						L = tril(M);
						precond_L = (D+L)*inv(D)*(D+L');
						precond_R = [];							
					elseif(strcmp(precond,'jacobi'))
						precond_L = spdiags(1./spdiags(M,0),0,size(M,1),size(M,2));
						precond_R = [];
					else
						precond_L = [];
						precond_R = [];
					end
				end
				z_tilde = pcg(M,rhs,tol_pcg,maxit_pcg,precond_L,precond_R,z_tilde);
			end

			
			zx = Dx*z_tilde;
			zy = Dy*z_tilde;
			Nx = zeros(nrows,ncols);
			Ny = zeros(nrows,ncols);
			Nz = zeros(nrows,ncols);
			Nx(imask) = fx*zx;
			Ny(imask) = fy*zy;
			if(orthographic)
				Nz(imask) = -1;
			else
				Nz(imask) = -u_tilde(imask).*zx-v_tilde(imask).*zy-1;
			end
			dz = sqrt(Nx.^2+Ny.^2+Nz.^2);
			N_old = N; % Store for convergence test
			N = cat(3,Nx./dz,Ny./dz,Nz./dz);
			
			% Auxiliary variables
			if(orthographic)
				z(imask) = z_tilde;
			else
				z(imask) = exp(z_tilde);
            end


            %%% [ADDITION] Update psi, and chi, before optimizing for the
            %%% lights. It is correct to do it here because rho is gotten
            %%% below from dz, and in the original code, dx is the updated
            %%% one, whereas Psi, chi are not (however, in the original
            %%% code, this was not a proble,, as psi is not used for
            %%% clamping rho. But only dz is. This however confirms that we
            %%% clamp rho with the most up-to-date variables? But Phi is
            %%% old...

            %%% Decision: don't clamp rho here, it is not theoretically
            %%% good

            if correct_psi_chi
                for ch = 1:nchannels
                    psi(:,:,ch) = reshape(shading_fcn(z_tilde,S(:,:,ch)),npix,nimgs);
                end
                chi = chi_fcn(psi);
            end
           
            % Update ambient
			for ch = 1:nchannels
				ambient_ch = ambient(:,:,ch);
				ambient_ch(imask) = ambient_tilde(:,ch);
				ambient(:,:,ch) = ambient_ch;
            end
            clear psich;

			if(orthographic)
				XYZ = cat(3,u_tilde,v_tilde,z);
			else
				XYZ = cat(3,z.*u_tilde./fx,z.*v_tilde./fy,z);
            end
			
			%%% Update S
			if(uncalibrated)
				NNt_11 = Nx(imask).^2;
				NNt_12 = Nx(imask).*Ny(imask);
				NNt_13 = Nx(imask).*Nz(imask);
				NNt_22 = Ny(imask).^2;
				NNt_23 = Ny(imask).*Nz(imask);
				NNt_33 = Nz(imask).^2;
				
				for ch = 1:nchannels
					Ich = I(:,:,ch);
					Wch = W_idx(:,:,ch);
					psich = psi(:,:,ch);
					rho_rep = repmat(rho_tilde(:,ch),[1 nimgs]);
					w(:,:,ch) = Wch.*w_fcn(reshape(r_fcn(rho_tilde(:,ch),psich(:),Ich(:),Phi(:,ch),ambient_tilde(:,ch),Wch(:)),npix,nimgs));
					clear psich
					% Aliases
					w_rho_chi = bsxfun(@times,w(:,:,ch).*rho_rep.*chi(:,:,ch),transpose(Phi(:,ch)));
					w_rho2_chi2 = bsxfun(@times,w_rho_chi.*rho_rep.*chi(:,:,ch),transpose(Phi(:,ch)));
					clear rho_rep
					
					% Construct second members
					w_rho_chi_I = w_rho_chi.*(bsxfun(@minus,Ich,ambient_tilde(:,ch)));
					clear w_rho_chi
					w_rho_chi_I_N = [sum(bsxfun(@times,w_rho_chi_I,Nx(imask)),1);
									sum(bsxfun(@times,w_rho_chi_I,Ny(imask)),1);...
									sum(bsxfun(@times,w_rho_chi_I,Nz(imask)),1)];
									
					sec_mb =  reshape(w_rho_chi_I_N+3*npix*stabilizer*transpose(S(:,:,ch)),[3*nimgs,1]);
					clear w_rho_chi_I w_rho_chi_I_N
					
					% Construct matrices 
					mat_i_11 = sum(bsxfun(@times,w_rho2_chi2,NNt_11),1)+3*npix*stabilizer;
					mat_i_12 = sum(bsxfun(@times,w_rho2_chi2,NNt_12),1);
					mat_i_13 = sum(bsxfun(@times,w_rho2_chi2,NNt_13),1);
					mat_i_22 = sum(bsxfun(@times,w_rho2_chi2,NNt_22),1)+3*npix*stabilizer;
					mat_i_23 = sum(bsxfun(@times,w_rho2_chi2,NNt_23),1);
					mat_i_33 = sum(bsxfun(@times,w_rho2_chi2,NNt_33),1)+3*npix*stabilizer;
					clear w_rho2_chi2
					rows_mat_i = [1:3:3*nimgs-2 1:3:3*nimgs-2 1:3:3*nimgs-2 2:3:3*nimgs-1 2:3:3*nimgs-1 2:3:3*nimgs-1 3:3:3*nimgs 3:3:3*nimgs 3:3:3*nimgs];
					cols_mat_i = [1:3:3*nimgs-2 2:3:3*nimgs-1 3:3:3*nimgs 1:3:3*nimgs-2 2:3:3*nimgs-1 3:3:3*nimgs 1:3:3*nimgs-2 2:3:3*nimgs-1 3:3:3*nimgs];
					mat_i = sparse(rows_mat_i,cols_mat_i,[mat_i_11 mat_i_12 mat_i_13 mat_i_12 mat_i_22 mat_i_23 mat_i_13 mat_i_23 mat_i_33]); 
					
					S(:,:,ch) = transpose(reshape(mat_i\sec_mb,[3 nimgs]));
					
					% Projection on the unit sphere
					norm_S(:,ch) = sqrt(sum(S(:,:,ch).^2,2));
					S(:,:,ch) = bsxfun(@rdivide,S(:,:,ch),norm_S(:,ch));
					psi(:,:,ch) = reshape(shading_fcn(z_tilde,S(:,:,ch)),npix,nimgs);
				end
				% update chi
				chi = chi_fcn(psi);
			end
			
			% Intensities update
			if(semi_calibrated)
				% Phi update
				for ch = 1:nchannels
					Ich = I(:,:,ch);
					Wch = W_idx(:,:,ch);
					psich = psi(:,:,ch);
					phich = Phi(:,ch);
					w(:,:,ch) = Wch.*max(0,w_fcn(reshape(r_fcn(rho_tilde(:,ch),psich(:),Ich(:),phich,ambient_tilde(:,ch),Wch(:)),npix,nimgs)));
					rho_psi_chi = psi(:,:,ch).*chi(:,:,ch).*repmat(rho_tilde(:,ch),[1 nimgs]);
					Phi(:,ch) = transpose(((sum(w(:,:,ch).*bsxfun(@minus,I(:,:,ch),ambient_tilde(:,ch)).*rho_psi_chi,1))+stabilizer*npix*Phi(:,ch)')./(sum(w(:,:,ch).*(rho_psi_chi).^2,1)+npix*stabilizer));
				end			
			end
			clear Ich psich rho_psi_chi		

			% Convergence test AND UPDATING psi
			energy_new = 0;
			for ch = 1:nchannels
				psi(:,:,ch) = reshape(shading_fcn(z_tilde,S(:,:,ch)),npix,nimgs);
				Ich = I(:,:,ch);
				Wch = W_idx(:,:,ch);
				psich = psi(:,:,ch);
				energy_new = energy_new+J_fcn(rho_tilde(:,ch),psich(:),Ich(:),Phi(:,ch),ambient_tilde(:,ch),Wch(:));
			end
			energy_new = energy_new./(npix*nimgs*nchannels);
			relative_diff = abs(energy_new-energy)./energy_new;
			
			normal_diff = rad2deg(real(acos(sum(N_old.*N,3))));
			normal_residual = median(normal_diff(imask));
			
			diverged = energy_new>energy;
			
			disp(sprintf('== it. %d - energy : %.6f - energy relative diff : %.6f - normal median variation: %.2f',it,energy_new,relative_diff,normal_residual));
			energy = energy_new;
			tab_nrj(it+1) = energy;
			
			% Display current result and convergence tests
			if(display)
				figure(hfig1);
				surfl(-XYZ(:,:,1),-XYZ(:,:,2),-XYZ(:,:,3),[0 90]);
				shading flat;
				colormap gray
				view(170,30);
				axis ij
				axis image
				title('Shape')

				figure(hfig2)
				semilogy(0:it,tab_nrj(1:it+1),'Linewidth',4)
				title('Energy')

				figure(hfig3)
				rho_disp = rho;
				if(nchannels==1)
					imagesc(rho_disp)
					colormap gray
					colorbar
				else
					rho_disp = uint8(255*rho_disp/max(rho_disp(:)));
					imagesc(rho_disp)					
				end
				axis image
				axis off
				title('Albedo')

				figure(hfig4)
				Ndisp = N;
				Ndisp(:,:,3) = -Ndisp(:,:,3);
				Ndisp = 0.5*(Ndisp+1);
				imagesc(Ndisp)
				axis image
				axis off
				title('Normals')
				
				figure(hfig5)
				imagesc(normal_diff,[0 10])
				axis image
				axis off
				colorbar
				title('Variation of normals between two iterations (in degrees)')			
				
				if(do_ambient)
					figure(hfig6)				
					if(nchannels==1)
						imagesc(ambient.*max_I)
						colormap gray
						colorbar
					else
						amb_disp = ambient-min(ambient(:));
						amb_disp = uint8(255*amb_disp/max(amb_disp(:)));
						imagesc(amb_disp)					
					end
					axis image
					axis off
					title('Ambient')
				end
							
				drawnow	
			end
			disp(' ');
% 			if((it>1) & (relative_diff<tol))
% 				disp('LAMBERTIAN CONVERGENCE: LOW ENERGY EVOLUTION. STARTING SPECULAR');
% 				start_specular_optimization = 1;
%                 if do_specular==0
%                     break;
%                 end
% 			end
% 			if((it>1) & (normal_residual<tol_normals))
% 				disp('CONVERGENCE: LOW SHAPE EVOLUTION');
% 				start_specular_optimization = 1;
%                 break;
% 			end
% 			if((it>1) & diverged)
% 				disp('STOPPED: DIVERGENT BEHAVIOR DETECTED');
% 				break;
% 			end
% 			if((it>1) & diverged)
% 				disp('STOPPED: DIVERGENT BEHAVIOR DETECTED');
% 				break;
% 			end
			if(it==maxit)
				disp('STOPPED: MAX NUMBER OF ITERATIONS REACHED');
				break;
			end
			
		end		
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% Update final result by interpolation
		if(ratio>1)
			disp('upscaling for the next level in the pyramid...');
			z(mask==0) = NaN;
			z=inpaint_nans(z);
			z_final = imresize(z,[nrows_ref ncols_ref]);
			rho_final = imresize(rho,[nrows_ref ncols_ref]);
			ambient_final = imresize(ambient,[nrows_ref ncols_ref]);
            shininess_final = imresize(shininess, [nrows_ref ncols_ref]);
            rho_S_final = imresize(rho_S, [nrows_ref ncols_ref]);
		end

	end % Scales
	
	
	%% Back-scale albedo and ambient light for consistency with the original images
	ambient = ambient.*max_I;
	Phi = Phi.*max_I;
	if(display)
		close(hfig1);
		close(hfig2);
		close(hfig3);
		close(hfig4);
		close(hfig5);
		if(do_ambient)
			close(hfig6);
		end
	end
	
	%% Fill non-estimated values with NaNs
	NaN_mask = NaN*mask;
	NaN_mask(mask>0) = 1;
	XYZ = bsxfun(@times,NaN_mask,XYZ);
	N = bsxfun(@times,NaN_mask,N);
	rho = bsxfun(@times,NaN_mask,rho);
	ambient = bsxfun(@times,NaN_mask,rho);
	
    disp("Add back support for residual weighting inside of specular!")
	disp(' ');
end	


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
	

% Functions for computing the gradient operator on non-rectangular domains
function [M,Dyp,Dym,Dxp,Dxm] = make_gradient(mask)

	% Compute forward (Dxp and Dyp, p is for plus) and backward (Dxm and Dym, m is for minus) operators
	[Dyp,Dym,Dxp,Dxm,Sup,Sum,Svp,Svm,Omega,index_matrix,imask] = gradient_operators(mask);  % all the D ... are size # valid pixels x # valid pixels
	[nrows,ncols] = size(mask);

	% When there is no bottom neighbor, replace by backward (or by 0 if no top)
	Dy = Dyp;   % size # valid pixels x # valid pixels
	no_bottom = find(~Omega(:,:,1));
	no_bottom = nonzeros(index_matrix(no_bottom));
	Dy(no_bottom,:) = Dym(no_bottom,:);

	% Same for the x direction (right / left)
	Dx = Dxp;   % size # valid pixels x # valid pixels
	no_right = find(~Omega(:,:,3));
	no_right = nonzeros(index_matrix(no_right));
	Dx(no_right,:) = Dxm(no_right,:);

	M = sparse([],[],[],2*size(Dx,1),size(Dx,2),2*length(imask));   % this is what is ouptutted
	M(1:2:end-1,:) = Dx;    % we pack both Dx and Dy into the same matrix, but then recover them
	M(2:2:end,:) = Dy;
end

function [Dup,Dum,Dvp,Dvm,Sup,Sum,Svp,Svm,Omega,index_matrix,imask] = gradient_operators(mask)

	[nrows,ncols] = size(mask);
	Omega_padded = padarray(mask,[1 1],0);

	% Pixels who have bottom neighbor in mask
	Omega(:,:,1) = mask.*Omega_padded(3:end,2:end-1);
	% Pixels who have top neighbor in mask
	Omega(:,:,2) = mask.*Omega_padded(1:end-2,2:end-1);
	% Pixels who have right neighbor in mask
	Omega(:,:,3) = mask.*Omega_padded(2:end-1,3:end);
	% Pixels who have left neighbor in mask
	Omega(:,:,4) = mask.*Omega_padded(2:end-1,1:end-2);
	

	imask = find(mask>0);   % a vector of indices
	index_matrix = zeros(nrows,ncols);
	index_matrix(imask) = 1:length(imask);

	% Dv matrix, size # valid pixels x # valid pixels
	% When there is a neighbor on the right : forward differences
	idx_c = find(Omega(:,:,3)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc,yc+1));
	indices_right = indices_right(:);
	II = indices_centre;
	JJ = indices_right;
	KK = ones(length(indices_centre),1);
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;-ones(length(indices_centre),1)];
	
	Dvp = sparse(II,JJ,KK,length(imask),length(imask));
	Svp = speye(length(imask));
	Svp = Svp(index_matrix(idx_c),:);

	% When there is a neighbor on the left : backward differences
	idx_c = find(Omega(:,:,4)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc,yc-1));
	indices_right = indices_right(:);
	II = [indices_centre];
	JJ = [indices_right];
	KK = [-ones(length(indices_centre),1)];
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;ones(length(indices_centre),1)];
	
	Dvm = sparse(II,JJ,KK,length(imask),length(imask));
	Svm = speye(length(imask));
	Svm = Svm(index_matrix(idx_c),:);



	% Du matrix: size # valid pixels x # valid pixels
	% When there is a neighbor on the bottom : forward differences
	idx_c = find(Omega(:,:,1)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc+1,yc));
	indices_right = indices_right(:);
	II = indices_centre;
	JJ = indices_right;
	KK = ones(length(indices_centre),1);
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;-ones(length(indices_centre),1)];
	
	Dup = sparse(II,JJ,KK,length(imask),length(imask));
	Sup = speye(length(imask));
	Sup = Sup(index_matrix(idx_c),:);

	% When there is a neighbor on the top : backward differences
	idx_c = find(Omega(:,:,2)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc-1,yc));
	indices_right = indices_right(:);
	II = [indices_centre];
	JJ = [indices_right];
	KK = [-ones(length(indices_centre),1)];
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;ones(length(indices_centre),1)];
	
	Dum = sparse(II,JJ,KK,length(imask),length(imask));
	Sum = speye(length(imask));
	Sum = Sum(index_matrix(idx_c),:);	

end


function [sortedA,sortIndex,sortIndex_orig] = sort_linear_index(A,sortDim,sortOrder)
%#SORT_LINEAR_INDEX   Just like SORT, but returns linear indices

  sizeA = size(A);  %# Get the matrix size
  if nargin < 2
    sortDim = find(sizeA > 1,1);  %# Define sortDim, if necessary
  end
  if nargin < 3
    sortOrder = 'ascend';  %# Define sortOrder, if necessary
  end
  [sortedA,sortIndex] = sort(A,sortDim,sortOrder);  %# Sort the matrix
  sortIndex_orig = sortIndex;
  [subIndex{1:numel(sizeA)}] = ...  %# Create a set of matrix subscripts
     ind2sub(sizeA,reshape(1:prod(sizeA),sizeA));
  subIndex{sortDim} = sortIndex;  %# Overwrite part of the subscripts with
                                  %#   the sort indices
  sortIndex = sub2ind(sizeA,subIndex{:});  %# Find the linear indices

end
