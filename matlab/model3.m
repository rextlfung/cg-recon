%% Model 3: Sensitivity maps, FT, and undersampling
classdef model3
    properties
        M
        N
        Ncoils
        smaps
    end

    methods
        %% Constructor
        function obj = model3()
            obj.M = 256;
            obj.N = 256;
            obj.Ncoils = 8;
            
            smaps = zeros(obj.M, obj.N, obj.Ncoils);
            ramp = repmat(linspace(0,1,obj.N), [obj.M,1]);
            for ncoil = 1:obj.Ncoils
                smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/obj.Ncoils, 'crop');
            end
            obj.smaps = smaps;
        end

        %% Forward model
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: multicoil k-space vector (M*N*Ncoils x 1)
        function k_vec = forward(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.M,obj.N]);
        
            % Create Ncoils copies of the image
            img = repmat(img, [1,1,obj.Ncoils]);
        
            % sensitivity maps
            img_coils = obj.smaps .* img;
        
            k_coils = ifftshift(fft2(fftshift(img_coils)));
        
            % undersampling mask
            mask = ones(obj.M,obj.N);
            mask(:,1:2:end) = 0; % skip every other line
            k_coils = mask .* k_coils;
            
            % collapse dimensions one at a time
            %k = reshape(k_coils,[M*N, Ncoils]);
            k_vec = k_coils(:);
        end

        %% Adjoint of model 3
        % Input argument: multicoil k-space vector (M*N*Ncoils x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = adjoint(obj, k_vec)
            % Reshape k-space vector into multicoil 2D matrices
            k_coils = reshape(k_vec, [obj.M,obj.N,obj.Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            img_coils = ifftshift(ifft2(fftshift(k_coils)));
        
            % weight each coil image by the conj of their smap
            img = conj(obj.smaps) .* img_coils;
        
            % Sum back to a single image
            img = sum(img,3);
        
            % Vectorize
            img_vec = img(:);
        end

        %% B = A_adj * A of model 3
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
            mask = ones(obj.M,obj.N);
            mask(:,1:2:end) = 0; % skip every other line
            k_coils = mask .* k_coils;
            
            % !! Commented out as it's redundant
            % % collapse dimensions one at a time
            % %k = reshape(k_coils,[M*N, Ncoils]);
            % k_vec = k_coils(:);
            % 
            % % Reshape k-space vector into multicoil 2D matrices
            % k_coils = reshape(k_vec, [M,N,Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            img_coils = ifftshift(ifft2(fftshift(k_coils)));
        
            % !! Commented out as it's redundant
            % Divide by sensitivity maps and get back to Ncoil copies of single coi
            % smaps = zeros(M, N, Ncoils);
            % ramp = repmat(linspace(0,1,N), [M,1]);
            % for ncoil = 1:Ncoils
            %     smaps(:,:,ncoil) = imrotate(ramp, 360*ncoil/Ncoils, 'crop');
            % end
        
            % weight each coil image by the conj of their smap
            img = conj(obj.smaps) .* img_coils;
        
            % Sum back to a single image
            img = sum(img,3);
        
            % Vectorize
            img_vec = img(:);
        end
    end
end