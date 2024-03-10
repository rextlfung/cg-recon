%% Model 4: Smaps, FT, 2x us + pf, and L2-regularization on TV(x)
classdef model5
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
        function obj = model5(smaps, sample_mask, lambda)
            obj.M = size(smaps, 1);
            obj.N = size(smaps, 2);
            obj.Ncoils = size(smaps, 3);
            obj.smaps = smaps;
            obj.sample_mask = sample_mask;
            obj.lambda = lambda;
        end

        %% Forward model 5
        % Input vector: img as a vector (M*N x 1)
        % Output vector: ((M*N*Ncoils + M*N) x 1)
        %   multicoil k-space vector (M*N*Ncoils x 1)
        %   regularizer output vector (M*N x 1)
        function out_vec = forward(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            % Create Ncoils copies of the image
            imgs = repmat(img, [1,1,obj.Ncoils]);
        
            % sensitivity maps
            img_coils = obj.smaps .* imgs;
        
            k_coils = ifftshift(fft2(fftshift(img_coils)));
        
            % undersampling mask
            k_coils = obj.sample_mask .* k_coils;
            
            % collapse dimensions one at a time
            %k = reshape(k_coils,[M*N, Ncoils]);
            k_vec = k_coils(:);

            % TV regularizer output
            TV_filt_x = [-1; 0; 1;];
            TV_filt_y = TV_filt_x.';
            img_tvx = conv2(img, TV_filt_x, 'same');
            img_tvy = conv2(img, TV_filt_y, 'same');

            % Multiply by lambda and flatten into vector
            reg_vec = sqrt(obj.lambda) .* [img_tvx(:); img_tvy(:)];

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

            % Extract TV img vectors from regularization vector
            reg_vec = sqrt(obj.lambda) .* reg_vec;
            img_tvx = reshape(reg_vec(1:obj.M*obj.N), [obj.M, obj.N]);
            img_tvy = reshape(reg_vec((obj.M*obj.N + 1):end), [obj.M, obj.N]);

            % Apply adjoint of TV operator to recover image
            TV_filt_x = [-1; 0; 1;];
            TV_filt_y = TV_filt_x.';
            img = conv2(img_tvx, flip(TV_filt_x), 'same');
            img = img + conv2(img_tvy, flip(TV_filt_y), 'same');

            % Reshape k-space vector into multicoil 2D matrices
            k_coils = reshape(k_vec, [obj.M,obj.N,obj.Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            img_coils = ifftshift(ifft2(fftshift(k_coils)));
        
            % weight each coil image by the conj of their smap,
            % then sum back into the same image
            img = img + sum(conj(obj.smaps) .* img_coils, 3);
        
            % Vectorize and add to image obtained from regularizer
            img_vec = img(:);
        end

        %% B = A_adj * A of model 4
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = both(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            % Create Ncoils copies of the image
            imgs = repmat(img, [1,1,obj.Ncoils]);
        
            % sensitivity maps
            img_coils = obj.smaps .* imgs;
        
            k_coils = ifftshift(fft2(fftshift(img_coils)));
        
            % undersampling mask
            k_coils = obj.sample_mask .* k_coils;
            
            % collapse dimensions one at a time
            %k = reshape(k_coils,[M*N, Ncoils]);
            k_vec = k_coils(:);

            % TV regularizer output
            TV_filt_x = [-1; 0; 1;];
            TV_filt_y = TV_filt_x.';
            img_tvx = conv2(img, TV_filt_x, 'same');
            img_tvy = conv2(img, TV_filt_y, 'same');

            % Multiply by lambda and flatten into vector
            reg_vec = sqrt(obj.lambda) .* [img_tvx(:); img_tvy(:)];

            % % vertically stack column vectors
            % out_vec = [k_vec; reg_vec];
            % 
            % %% backward
            % % Separate concatenated vectors
            % k_vec = in_vec(1:obj.M*obj.N*obj.Ncoils);
            % reg_vec = in_vec((obj.M*obj.N*obj.Ncoils + 1):end);

            % Extract TV img vectors from regularization vector
            reg_vec = sqrt(obj.lambda) .* reg_vec;
            img_tvx = reshape(reg_vec(1:obj.M*obj.N), [obj.M, obj.N]);
            img_tvy = reshape(reg_vec((obj.M*obj.N + 1):end), [obj.M, obj.N]);

            % Apply adjoint of TV operator to recover image
            TV_filt_x = [-1; 0; 1;];
            TV_filt_y = TV_filt_x.';
            img = conv2(img_tvx, flip(TV_filt_x), 'same');
            img = img + conv2(img_tvy, flip(TV_filt_y), 'same');

            % Reshape k-space vector into multicoil 2D matrices
            k_coils = reshape(k_vec, [obj.M,obj.N,obj.Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            img_coils = ifftshift(ifft2(fftshift(k_coils)));
        
            % weight each coil image by the conj of their smap,
            % then sum back into the same image
            img = img + sum(conj(obj.smaps) .* img_coils, 3);
        
            % Vectorize and add to image obtained from regularizer
            img_vec = img(:);
        end
    end
end