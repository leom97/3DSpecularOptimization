clear; clc;
name = "backpack";
path = "..\OpenGL\data\models\"+name+"\synthetic\non_lambertian";

%% Read from conf.txt the important parameters
fid = fopen(path+"\"+"conf.txt");
tline = fgetl(fid);
tline_N_frames = fgetl(fid);

tline = fgetl(fid);
tline_H = fgetl(fid);
tline_W = fgetl(fid);

tline = fgetl(fid);
tline_depth_mode = fgetl(fid);

tline = fgetl(fid);
tline_near = fgetl(fid);
tline_far = fgetl(fid);

tline = fgetl(fid);
tline_fx = fgetl(fid);
tline_fy = fgetl(fid);
tline_cx = fgetl(fid);
tline_cy = fgetl(fid);

fclose(fid);

N_frames = str2double(tline_N_frames);
H = str2double(tline_H);
W = str2double(tline_W);
depth_mode = string(tline_depth_mode);
near = str2double(tline_near);
far = str2double(tline_far);
fx = str2double(tline_fx);
fy = str2double(tline_fy);
cx = str2double(tline_cx);
cy = str2double(tline_cy);

K = eye(3);
K(1,1) = fx; K(2,2) = fy; K(1,3) = cx; K(2,3) = cy;

%% Read camera pose
% (to transform the normals, idiot)
fid = fopen(path+"\"+"0_camera_pose_and_model.txt");
tline = fgetl(fid);
tline = fgetl(fid);

T_WC = eye(4);
for i=1:4
    for j=1:4
        T_WC(i,j) = str2double(fgetl(fid));
    end
end

T_CW = inv(T_WC);
R_CW = T_CW(1:3,1:3);

for i=1:9
    tline = fgetl(fid);
end

model=zeros(4,4);
for i=1:4
    for j=1:4
        model(i,j) = str2double(fgetl(fid));
    end
end
fclose(fid);

%% Per frame logic

mask = zeros(H,W);
I = zeros(H,W,3,N_frames);
L = zeros(N_frames, 3);

for i=1:N_frames
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % HDR images
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    t = readtable(path + "\" + (i-1) +"_HDRAlpha.txt", 'ReadVariableNames', false);
    HDRA_vector = t.Variables;
    clear t;

    % Get image channels
    ri = 1:4:(4*H*W);
    r = HDRA_vector(ri);
    gi = 2:4:(4*H*W);
    g = HDRA_vector(gi);
    bi = 3:4:(4*H*W);
    b = HDRA_vector(bi);
    ai = 4:4:(4*H*W);
    a = HDRA_vector(ai);

    % Form HDR image
    rho = zeros(H, W, 3);
    rho(:,:,1) = flipud(reshape(r, W, H)');
    rho(:,:,2) = flipud(reshape(g, W, H)');
    rho(:,:,3) = flipud(reshape(b, W, H)');
    
    I(:,:,:,i) = rho;
    mask = logical(flipud(reshape(a, W, H)'));
    
    % Show the image
    imshow(rho)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Lights
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fid = fopen(path+"\"+(i-1)+"_light_vector.txt");
    tline = fgetl(fid);
    li = zeros(3,1);
    for j=1:3
        tline = fgetl(fid);
        li(j)=str2double(tline); % Light vector in world frame, pointing to the object, OpenGL conventions
    end
    fclose(fid);
    
    li = R_CW * li; % light vector in camera frame
    disp(li);
    li(3) = -li(3); % pointing to the object (because z is flipped)
    li = -li;   % pointing away from object
    
    L(i, :) = li.';
    
end

S = L;

%%  Depth images (problem: not at very high resolution!)

blender = 0;
if blender==0
    t = readtable(path + "\" + 0 +"_depth_map_" + depth_mode + ".txt", 'ReadVariableNames', false);
    depth_vector = t.Variables;
    clear t;

    % Note, the depth are positive!
    z = flipud(reshape(depth_vector, W, H)');
    if depth_mode == "standard"
        z = z * 2.0 - 1.0; 
        z =(2.0 * near * far) ./ (far + near - z .* (far - near));
    elseif depth_mode == "reverse"
        z = (near * far) ./ (near + z * (far - near));
    else
        disp("Depth mode not recognized...");
        return
    end
    
    z(mask==0) = NaN;
else
    t = readtable(path + "\" +"depth_blender.txt", 'ReadVariableNames', false);
    depth_vector = t.Variables;
    clear t;

    % Note, the depth are positive!
    z = flipud(reshape(depth_vector, W, H)');
    
    % Some Blender specific postprocessing (mask is off ever so slightly..)
    z(z>far) = NaN;
    z(mask==0) = NaN;
    z = inpaint_nans(z);
    z(mask==0) = NaN;
    
end

%% Display the point cloud

[nrows,ncols] = size(mask);
[uu,vv] = meshgrid(1:ncols,1:nrows);
u_tilde = (uu - cx); 
v_tilde = flipud(vv - cy);
z_mask = inpaint_nans(z);
% z_mask = imguidedfilter(z_mask);
z_mask(mask==0)=NaN;
XYZ = cat(3,z_mask .* u_tilde./fx,z_mask.*v_tilde./fy,z_mask);
surfl(XYZ(:,:,1),XYZ(:,:,2),XYZ(:,:,3));
shading flat;
colormap gray
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

%% Normals (for ground truth)

t = readtable(path + "\" + 0 +"_normals" + ".txt", 'ReadVariableNames', false);
normals_vector = t.Variables;
clear t;

xi = 1:3:(3*H*W);
xN = normals_vector(xi);
yi = 2:3:(3*H*W);
yN = normals_vector(yi);
zi = 3:3:(3*H*W);
zN = normals_vector(zi);

NMat = [xN,yN,zN]';
xN = NMat(1,:);
yN = NMat(2,:);
zN = NMat(3,:);

% Processing in OpenGL world frame
N = zeros(H, W, 3);
N(:,:,1) = (flipud(reshape(xN, W, H)'))*2-1; % *2 - 1 because of the OpenGL processing
N(:,:,2) = (flipud(reshape(yN, W, H)'))*2-1;
N(:,:,3) = (flipud(reshape(zN, W, H)')*2-1);
N = N./sqrt(sum(N.^2,3));

% Processing in camera frame
xN = reshape(N(:,:,1), H*W, 1);
yN = reshape(N(:,:,2), H*W, 1);
zN = reshape(N(:,:,3), H*W, 1);
NMat = R_CW * [xN,yN,zN]';
N(:,:,1) = reshape(NMat(1,:), H,W);
N(:,:,2) = reshape(NMat(2,:), H,W);
N(:,:,3) = -reshape(NMat(3,:), H,W);    % NB. Flipping the normal is done in the frame where the depth is flipped, idiot (so, in the camera frame)

% Check for the normals (they give the inner pointing normals)
Ncheck = zeros(size(N));
[Ncheck(:,:,1), Ncheck(:,:,2), Ncheck(:,:,3)] = surfnorm(XYZ(:,:,1),XYZ(:,:,2),XYZ(:,:,3));

%% I have to plot the normals
quiver3(XYZ(:,:,1),XYZ(:,:,2),XYZ(:,:,3), N(:,:,1), N(:,:,2), N(:,:,3), 7);
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

%% Albedos

% Diffuse
t = readtable(path + "\" + 0 +"_diffuse_albedo.txt", 'ReadVariableNames', false);
HDRA_vector = t.Variables;
clear t;

% Get image channels
ri = 1:4:(4*H*W);
r = HDRA_vector(ri);
gi = 2:4:(4*H*W);
g = HDRA_vector(gi);
bi = 3:4:(4*H*W);
b = HDRA_vector(bi);

% Form HDR image
rho = zeros(H, W, 3);
rho(:,:,1) = flipud(reshape(r, W, H)');
rho(:,:,2) = flipud(reshape(g, W, H)');
rho(:,:,3) = flipud(reshape(b, W, H)');

% Specular
t = readtable(path + "\" + 0 +"_specular_albedo.txt", 'ReadVariableNames', false);
HDRA_vector = t.Variables;
clear t;

% Get image channels
ri = 1:4:(4*H*W);
r = HDRA_vector(ri);
gi = 2:4:(4*H*W);
g = HDRA_vector(gi);
bi = 3:4:(4*H*W);
b = HDRA_vector(bi);

% Form HDR image
rho_S = zeros(H, W, 3);
rho_S(:,:,1) = flipud(reshape(r, W, H)');
rho_S(:,:,2) = flipud(reshape(g, W, H)');
rho_S(:,:,3) = flipud(reshape(b, W, H)');

%% Display albedos

imshow(rho)
imshow(rho_S)

%% Lights check
% 
% view(3)
% hold on
% im = 7;
% warp(XYZ(:,:,1),XYZ(:,:,2),XYZ(:,:,3), I(:,:,:,im));
% 
% x = arrow3([0,0,0], L(im,:));
% 
% axis equal
% xlabel('x')
% ylabel('y')
% zlabel('z')
% 
% hold off

%% Post processing

% Quantities useful for debugging, but they are defined in the camera
% system with y pointing upwards, x right, z toward the negatives
N_display = N;
S_display = L;

% But the real quantities... are in a right handed coordinate system
N(:,:,2) = -N(:,:,2);
S(:,2) = -S(:,2);

%% Save
save(name+"_non_lamb_spec_test.mat",'I','z','N','S','rho','rho_S','K','mask','N_display','S_display', 'XYZ');