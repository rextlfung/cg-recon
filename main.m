%% Practice using pcg() to reconstruct images
% Rex Fung
% Feb 29, 2024
clear; close all;

%% Load in phantom data as ground 
img_gt = phantom('Modified Shepp-Logan');
img_gt = img_gt .* exp(1i * img_gt); % make complex
img_gt = img_gt.'; % transpose for Jeff's im()
[M, N] = size(img_gt);

%% Visualize
figure; tiledlayout('flow','TileSpacing','tight');
nexttile; im(abs(img_gt)); title('abs'); colorbar;
nexttile; im(angle(img_gt)); title('angle'); colorbar;
nexttile; im(real(img_gt)); title('real'); colorbar;
nexttile; im(imag(img_gt)); title('imag'); colorbar;
sgtitle('Ground Truth')

%% Model 1: Only Fourier Transform
% Solve Ax = k where A is composed of:
% 1. fft2()
% 2. fftshift()
% 3. reshape k-space matrix into vector
k = model1(img_gt);
img_recon = pcg(@model1,k);
img_recon = reshape(img_recon, [M, N]);

figure; compareImages(img_gt, img_recon)

%% Model 2: FT and undersample in k-space
% Solve Ax = k where A is composed of:
% 1. fft2()
% 2. fftshift()
% 3. undersample k-space by skipping every other ky line
% 4. reshape k-space matrix into vector
k = model2(img_gt);
img_recon = pcg(@model2,k);
img_recon = reshape(img_recon, [M, N]);

figure; compareImages(img_gt, img_recon)

%% Model 3: FT, undersample, multicoil
% Solve Ax = k where A is composed of:
% 1. Multiplication of image by each sensitivity map --> Ncoil images
% 2. For each coil
%    1. fft2()
%    2. fftshift()
%    3. undersample k-space by skipping every other ky line
%    4. reshape k-space matrix into vector
% 3. stack all k-space vectors into a single vector
k = model3(img_gt);
img_recon = pcg(@model3,k);
img_recon = reshape(img_recon, [M, N]);

figure; compareImages(img_gt, img_recon)