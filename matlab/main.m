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

%% Turn image into a vector for pcg
img_vec = img_gt(:);

%% Model 1: Only Fourier Transform
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. Reshaping image vector x into an image
% 2. Fourier transforming into k-space
% 3. Reshaping k-space into a vector
model = model1();
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);
figure; compareImages(img_gt, img_recon);

%% Model 2: FT and undersample in k-space
% Solve A^H * A * x = A^H * k where A is composed of:
% 1. fft2()
% 2. fftshift()
% 3. undersample k-space by skipping every other ky line
% 4. reshape k-space matrix into vector

% Sampling mask
sample_mask = ones(M,N);
sample_mask(:,1:2:end) = 0; % skip every other line

model = model2(sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);
figure; compareImages(img_gt, img_recon);

%% Model 3: FT, undersample, and multicoil
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
sample_mask = ones(M,N);
sample_mask(:,1:2:end) = 0; % skip every other line

model = model3(smaps, sample_mask);
k_vec = model.forward(img_vec);
img_recon_vec = pcg(@model.both, model.adjoint(k_vec));
img_recon = reshape(img_recon_vec, [M,N]);
figure; compareImages(img_gt, img_recon);

%% Model 4: FT, undersample, multicoil, and regularization