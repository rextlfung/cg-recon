%% Practice using pcg() to reconstruct images
% Rex Fung
% Feb 29, 2024
clear; close all;

%% Load in phantom data as ground 
img_gt = phantom('Modified Shepp-Logan');
img_gt = img_gt .* exp(1i * 2*pi * img_gt); % make complex (add phase)
img_gt = img_gt.'; % transpose for Jeff's im()
[M, N] = size(img_gt);
img_vec = img_gt(:); % turn into a vector for pcg()

%% Visualize
figure; tiledlayout('flow','TileSpacing','tight');
nexttile; im(abs(img_gt)); title('abs'); colorbar;
nexttile; im(angle(img_gt)); title('angle'); colorbar;
nexttile; im(real(img_gt)); title('real'); colorbar;
nexttile; im(imag(img_gt)); title('imag'); colorbar;
sgtitle('Ground Truth')

%% Model 1: Only Fourier Transform
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. Reshaping image vector x into an image
% 2. Fourier transforming into k-space
% 3. Reshaping k-space into a vector
model = model1();
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N]))));
figure; compareImages(img_gt, img_us, img_recon);

%% Model 2: FT and 2x undersample in k-space
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. fft2()
% 2. fftshift()
% 3. undersample k-space by skipping every other ky line
% 4. reshape k-space matrix into vector

% Sampling mask
sample_mask = zeros(M,N);
sample_mask(:,1:2:end) = 1; % sample every 2 lines

model = model2(sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N]))));
figure; compareImages(img_gt, img_us, img_recon);

%% Model 3: FT, 2x undersample, and multicoil
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. Reshaping image vector x into an image
% 2. Creating Ncoil copies of the image
% 3. Weighting each image by each coil's sensitivity map
% 4. Fourier transforming each coil image into k-space
% 5. Reshaping k-space into a vector

% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask
sample_mask = zeros(M,N);
sample_mask(:,1:2:end) = 1; % sample every 2 lines

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 3a: Model 3 but symmetric undersampling about ky
% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask
sample_mask = zeros(M,N);
sample_mask(:,1:2:N/2) = 1; % sample every 2 lines
sample_mask(:,(N/2+1):end) = flip(sample_mask(:,1:N/2),2);

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 3b: Model 3 but 3x undersampling
% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask
sample_mask = zeros(M,N);
sample_mask(:,1:3:end) = 1; % sample every 3 lines

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 3c: Model 3 but 5/8 partial fourier
% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask with partial fourier
pf = 6/8;
sample_mask = zeros(M,N);
sample_mask(:,1:2:N*pf) = 1; % sample every other line with pf

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 3d: Model 3 but 5/8 partial fourier in both kx and ky
% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask with partial fourier
pf = 6/8;
sample_mask = zeros(M,N);
sample_mask(1:N*pf,1:2:N*pf) = 1; % sample every other line with pf

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 3e: Model 3 but grid like sampling (CAIPI)
% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask
% sample every other location in kx, ky
sample_mask = zeros(M,N);
sample_mask(1:2:M,1:2:N) = 1; 
sample_mask(2:2:M,2:2:N) = 1;

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 3e: Model 3 but grid like sampling (CAIPI) and PF
% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask with pf in ky
% sample every other location in kx, ky
pf = 6/8;
sample_mask = zeros(M,N);
sample_mask(1:2:M,1:2:N*pf) = 1; 
sample_mask(2:2:M,2:2:N*pf) = 1;

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 3f: Model 3 but grid like sampling (CAIPI) and PF in both dirs
% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask with pf in ky
% sample every other location in kx, ky
pf = 6/8;
sample_mask = zeros(M,N);
sample_mask(1:2:M*pf,1:2:N*pf) = 1; 
sample_mask(2:2:M*pf,2:2:N*pf) = 1;

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);

img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)

%% Model 4: Smaps, FT, 2x us + pf, and L2-regularization
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. Reshaping image vector x into an image
% 2. Creating Ncoil copies of the image
% 3. Weighting each image by each coil's sensitivity map
% 4. Fourier transforming each coil image into k-space
% 5. Reshaping k-space into a vector
% 6. Multiplying x by regularization term lambda, then concatenating it to
% the output vector

% Multicoil images with sensitivity maps
Ncoils = 8;
smaps = zeros(M, N, Ncoils);
ramp = repmat(linspace(0,1,N), [M,1]);
for ncoil = 1:Ncoils
    smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
end

% Sampling mask
sample_mask = zeros(M,N);
sample_mask(:,1:2:end) = 1; % sample every 2 lines

% Regularization
lambda = 1e-6;

model = model4(smaps, sample_mask, lambda);
y = model.forward(img_vec);
% Set "observed" regularization output to 0 for L2-regularization
y((M*N*Ncoils + 1):end) = 0;
img_recon_vec = pcg(@model.both, model.adjoint(y));
img_recon = reshape(img_recon_vec, [M,N]);

k_vec = y(1:M*N*Ncoils);
img_us = ifftshift(ifft2(fftshift(reshape(k_vec, [M,N,Ncoils]))));
img_us = mean(img_us,3);
figure; compareImages(img_gt, img_us, img_recon)