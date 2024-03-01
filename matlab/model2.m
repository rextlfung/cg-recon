%% Model 2: FT and undersampling in k-space
% Input arguments: img (M x N)
% Output arguments: k-space vector (M*N x 1)
function k = model2(img)
    [M, N] = size(img);
    mask = ones(M,N);
    mask(:,1:2:end) = 0; % skip every other line

    k = fftshift(fft2(img));
    k = k .* mask;
    k = k(:);
end