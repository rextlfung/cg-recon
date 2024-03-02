%% Model 1: Fourier Transform
classdef model1
    properties
        M
        N
    end

    methods
        %% Constructor
        function obj = model1()
            obj.M = 256;
            obj.N = 256;
        end

        %% Forward model 1
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: k-space vector (M*N x 1)
        function k_vec = forward(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            k = ifftshift(fft2(fftshift(img)));
            
            % collapse dimensions one at a time
            %k = reshape(k_coils,[M*N, Ncoils]);
            k_vec = k(:);
        end

        %% Adjoint of model 1
        % Input argument: k-space vector (M*N x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = adjoint(obj, k_vec)
            % Reshape k-space vector into 2D matrix
            k = reshape(k_vec, [obj.M,obj.N]);
        
            % IFFT to get back images
            img = ifftshift(ifft2(fftshift(k)));
        
            % Vectorize
            img_vec = img(:);
        end

        %% B = A_adj * A of model 1
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = both(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            k = ifftshift(fft2(fftshift(img)));
        
            % IFFT to get back images
            img = ifftshift(ifft2(fftshift(k)));
        
            % Vectorize
            img_vec = img(:);
        end
    end
end