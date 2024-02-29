%% Model 1: Only Fourier transform
% Input arguments: img (M x N)
% Output arguments: k-space vector (M*N x 1)
function k = model1(img)
    k = fftshift(fft2(img));
    k = k(:);
end