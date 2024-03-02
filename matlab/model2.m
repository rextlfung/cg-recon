%% Model 2: FT and undersampling
classdef model2
    properties
        M
        N
        sample_mask
    end

    methods
        %% Constructor
        function obj = model2(sample_mask)
            obj.M = size(sample_mask, 1);
            obj.N = size(sample_mask, 2);
            obj.sample_mask = sample_mask;
        end

        %% Forward model 2
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: k-space vector (M*N x 1)
        function k_vec = forward(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            k = ifftshift(fft2(fftshift(img)));
        
            % undersampling mask
            k = obj.sample_mask .* k;
            
            % collapse dimensions one at a time
            %k = reshape(k_coils,[M*N, Ncoils]);
            k_vec = k(:);
        end

        %% Adjoint of model 2
        % Input argument: k-space vector (M*N x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = adjoint(obj, k_vec)
            % Reshape k-space vector into 2D matrix
            k = reshape(k_vec, [obj.M,obj.N]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back images
            img = ifftshift(ifft2(fftshift(k)));
        
            % Vectorize
            img_vec = img(:);
        end

        %% B = A_adj * A of model 2
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = both(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            k = ifftshift(fft2(fftshift(img)));
        
            % undersampling mask
            k = obj.sample_mask .* k;
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back images
            img = ifftshift(ifft2(fftshift(k)));
        
            % Vectorize
            img_vec = img(:);
        end
    end
end