%% Recon some brain images using CG-SENSE
clear; close all;

fn_ld = '../largeData/D_te1.mat'; % large 3D multicoil file
fn_sd = '../smallData/k_mc2d.mat'; % small 2D multicoil file
fn_ld_smaps = '../largeData/D_te1_mps.mat'; % sensitivity maps for 3D data

load(fn_ld);
[N1, N2, N3, Ncoils] = size(D_te1);

%% Get 3D sensitivity maps with PISCO
if isfile(fn_ld_smaps)
    load(fn_ld_smaps);
else
    addpath('~/github/pisco');
    hybrid_data = ifftshift(ifft(fftshift(D_te1),N3,3));
    smaps = zeros(size(D_te1));
    for n = 1:N3
        [map_n, eigvals] = PISCO_senseMaps_estimation(squeeze(hybrid_data(:,:,n,:)), [N1, N2]);
        smaps(:,:,n,:) = reshape(map_n,[N1, N2, 1, Ncoils]);
    end
    save(fn_ld_smaps, 'smaps');
end

%% Make 3D CAIPI sample mask
Ry = 2;
Rz = 3;
caipi_mask = zeros(N1,N2,N3);
for shift = 1:Ry    
    caipi_mask(:,shift:Ry:end,(1+Rz*(shift-1)):Ry*Rz:end) = 1;
end
figure; im(caipi_mask)
%% Model 3: FT, 2x undersample, and multicoil
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. Reshaping image vector x into an image
% 2. Creating Ncoil copies of the image
% 3. Weighting each image by each coil's sensitivity map
% 4. Fourier transforming each coil image into k-space
% 5. Reshaping k-space into a vector

% Undersample at various factors
for R = 6:6
    sample_mask = zeros(N1,N2);
    sample_mask(:,1:R:end) = 1; % sample every R lines
    sample_mask = caipi_mask;
    ksp_us = sample_mask .* D_te1;
    
    % recon
    model = model3_3d(smaps, sample_mask);
    img_recon_vec = pcg(@model.both, model.adjoint(ksp_us(:)));
    
    img_recon = reshape(img_recon_vec, [N1, N2, N3]); % rehsape vector into image
    
    img_us = toppe.utils.ift3(ksp_us);
    img_us = mean(img_us,4); % Average the coil images for naive recon
    
    img_gt = sum(toppe.utils.ift3(D_te1) .* conj(smaps), 4);
    figure;
    [MSE, SSIM] = compareImages(img_gt, img_us, img_recon);
    title = sprintf('Model 3: FT, %dx undersample, and multicoil. SSIM = %f', R, SSIM);
    sgtitle(title);
end