%% Model 4: Smaps, FT, 2x us + pf, and L2-regularization
classdef model4
    properties
        M
        N
        Ncoils
        smaps
        sample_mask
        lambda
    end

    methods
        %% Constructor
        function obj = model4(smaps, sample_mask, lambda)
            obj.M = size(smaps, 1);
            obj.N = size(smaps, 2);
            obj.Ncoils = size(smaps, 3);
            obj.smaps = smaps;
            obj.sample_mask = sample_mask;
            obj.lambda = lambda;
        end

        %% Forward model 4
        % Input vector: img as a vector (M*N x 1)
        % Output vector: ((M*N*Ncoils + M*N) x 1)
        %   multicoil k-space vector (M*N*Ncoils x 1)
        %   regularizer output vector (M*N x 1)
        function out_vec = forward(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            % Create Ncoils copies of the image
            img = repmat(img, [1,1,obj.Ncoils]);
        
            % sensitivity maps
            img_coils = obj.smaps .* img;
        
            k_coils = ifftshift(fft2(fftshift(img_coils)));
        
            % undersampling mask
            k_coils = obj.sample_mask .* k_coils;
            
            % collapse dimensions one at a time
            %k = reshape(k_coils,[M*N, Ncoils]);
            k_vec = k_coils(:);

            % regularizer output
            reg_vec = sqrt(obj.lambda) .* img_vec;

            % vertically stack column vectors
            out_vec = [k_vec; reg_vec];
        end

        %% Adjoint of model 4
        % Input vector: ((M*N*Ncoils + M*N) x 1)
        %   multicoil k-space vector (M*N*Ncoils x 1)
        %   regularizer output vector (M*N x 1)
        % Output vector: img as a vector (M*N x 1)
        function img_vec = adjoint(obj, in_vec)
            % Separate concatenated vectors
            k_vec = in_vec(1:obj.M*obj.N*obj.Ncoils);
            reg_vec = in_vec((obj.M*obj.N*obj.Ncoils + 1):end);

            % Calculate image from regularizer
            img_vec = conj(sqrt(obj.lambda)) .* reg_vec;

            % Reshape k-space vector into multicoil 2D matrices
            k_coils = reshape(k_vec, [obj.M,obj.N,obj.Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            img_coils = ifftshift(ifft2(fftshift(k_coils)));
        
            % weight each coil image by the conj of their smap
            img = conj(obj.smaps) .* img_coils;
        
            % Sum back to a single image
            img = sum(img,3);
        
            % Vectorize and add to image obtained from regularizer
            img_vec = img_vec + img(:);
        end

        %% B = A_adj * A of model 4
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = both(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            % Create Ncoils copies of the image
            img = repmat(img, [1,1,obj.Ncoils]);
        
            % sensitivity maps
            img_coils = obj.smaps .* img;
        
            k_coils = ifftshift(fft2(fftshift(img_coils)));
        
            % undersampling mask
            k_coils = obj.sample_mask .* k_coils;
            
            % % collapse dimensions one at a time
            % %k = reshape(k_coils,[M*N, Ncoils]);
            % k_vec = k_coils(:);
             
            % regularizer output
            reg_vec = sqrt(obj.lambda) .* img_vec;
            
            % % vertically stack column vectors
            % out_vec = [k_vec; reg_vec];
            % 
            % % Separate concatenated vectors
            % k_vec = in_vec(1:obj.M*obj.N*obj.Ncoils);
            % reg_vec = in_vec((obj.M*obj.N*obj.Ncoils + 1):end);
            
            % Calculate image from regularizer
            img_vec = conj(sqrt(obj.lambda)) .* reg_vec;
            
            % % Reshape k-space vector into multicoil 2D matrices
            % k_coils = reshape(k_vec, [obj.M,obj.N,obj.Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            img_coils = ifftshift(ifft2(fftshift(k_coils)));
        
            % weight each coil image by the conj of their smap
            img = conj(obj.smaps) .* img_coils;
        
            % Sum back to a single image
            img = sum(img,3);
        
            % Vectorize and add to image obtained from regularizer
            img_vec = img_vec + img(:);
        end
    end
end