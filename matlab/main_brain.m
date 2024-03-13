%% Recon some brain images using CG-SENSE
clear; close all;

fn_ld = '../largeData/D_te1.mat'; % large 3D multicoil file
fn_sd = '../smallData/k_mc2d.mat'; % small 2D multicoil file

if isfile(fn_sd)
    load(fn_sd)
    [N1, N2, Ncoils] = size(k_mc2d);
else
    load(fn);
    [N1, N2, N3, Ncoils] = size(D_te1);
    
    % Extract a slice to do 2D recon
    D_te1 = flip(D_te1,2); % Flip PE direction
    D_te1 = ifftshift(ifft(fftshift(D_te1),[],3));
    slice = 55;
    k_mc2d = squeeze(D_te1(:,:,slice,:));
end

%% Get sensitivity maps with PISCO
addpath('~/github/pisco');
[smaps, eigvals] = PISCO_senseMaps_estimation(k_mc2d, [N1, N2]);

%% Model 3: FT, 2x undersample, and multicoil
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. Reshaping image vector x into an image
% 2. Creating Ncoil copies of the image
% 3. Weighting each image by each coil's sensitivity map
% 4. Fourier transforming each coil image into k-space
% 5. Reshaping k-space into a vector

% Undersample at various factors
for R = 2:6
    sample_mask = zeros(N1,N2);
    sample_mask(:,1:R:end) = 1; % sample every R lines
    k_mc2d_us = sample_mask .* k_mc2d;
    k_vec = k_mc2d_us(:);
    
    model = model3(smaps, sample_mask);
    img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
    img_recon = reshape(img_recon_vec, [N1, N2]);
    
    img_us = ifftshift(ifft2(fftshift(k_mc2d_us)));
    img_us = mean(img_us,3); % Average the coil images for naive recon
    
    img_gt = sum(ifftshift(ifft2(fftshift(k_mc2d))) .* conj(smaps), 3);
    figure;
    [MSE, SSIM] = compareImages(img_gt, img_us, img_recon);
    title = sprintf('Model 3: FT, %dx undersample, and multicoil. SSIM = %f', R, SSIM);
    sgtitle(title);
end