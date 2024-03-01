%% Model 3: Sensitivity maps, FT, and undersampling
% Input arguments: img (M x N)
% Output arguments: k-space vector (M*N x 1)
function k = model3(img)
    % sensitivity maps
    [M, N] = size(img);
    Ncoils = 8;
    smaps = zeros(M, N, Ncoils);
    ramp = repmat(linspace(0,1,N), [M,1]);
    for ncoil = 1:Ncoils
        smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
    end
    img_coils = smaps .* img;

    k_coils = fftshift(fft2(img_coils));

    % undersampling mask
    mask = ones(M,N);
    mask(:,1:2:end) = 0; % skip every other line
    k_coils = mask .* k_coils;
    
    % collapse dimensions one at a time
    k = reshape(k_coils,[M*N, Ncoils]);
    k = k(:);
end