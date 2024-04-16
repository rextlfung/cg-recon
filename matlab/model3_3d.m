%% Model 3: Sensitivity maps, FT, and undersampling
classdef model3_3d
    properties
        Nx
        Ny
        Nz
        Ncoils
        smaps
        sample_mask
    end

    methods
        %% Constructor
        function obj = model3_3d(smaps, sample_mask)
            obj.Nx = size(smaps, 1);
            obj.Ny = size(smaps, 2);
            obj.Nz = size(smaps, 3);
            obj.Ncoils = size(smaps, 4);
            obj.smaps = smaps;
            obj.sample_mask = sample_mask;
        end

        %% Forward model 3
        % Input arguments: img as a vector (Nx*Ny*Nz x 1)
        % Output arguments: multicoil k-space vector (Nx*Ny*Nz*Ncoils x 1)
        function k_vec = forward(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.Nx,obj.Ny,obj.Nz]);
        
            % Create Ncoils copies of the image
            img = repmat(img, [1,1,1,obj.Ncoils]);
        
            % sensitivity maps
            img_coils = obj.smaps .* img;
        
            % Fourier transform
            data = fftshift(img_coils);
            for dim = 1:3
                data = fft(data,size(data,dim),dim);
            end
            k_coils = ifftshift(data);
        
            % undersampling mask
            k_coils = obj.sample_mask .* k_coils;
            
            % collapse dimensions one at a time
            %k = reshape(k_coils,[Nx*Ny*Nz, Ncoils]);
            k_vec = k_coils(:);
        end

        %% Adjoint of model 3
        % Input argument: multicoil k-space vector (M*N*Ncoils x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = adjoint(obj, k_vec)
            % Reshape k-space vector into multicoil 2D matrices
            k_coils = reshape(k_vec, [obj.Nx,obj.Ny,obj.Nz,obj.Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            data = fftshift(k_coils);
            for dim = 1:3
                data = ifft(data,size(data,dim),dim);
            end
            img_coils = ifftshift(data);
        
            % weight each coil image by the conj of their smap
            img = conj(obj.smaps) .* img_coils;
        
            % Sum along coils to get back to a single image
            img = sum(img,4);
        
            % Vectorize
            img_vec = img(:);
        end

        %% B = A_adj * A of model 3
        % Input arguments: img as a vector (M*N x 1)
        % Output arguments: img as a vector (M*N x 1)
        function img_vec = both(obj, img_vec)
            % Reshape img vector into image
            img = reshape(img_vec, [obj.Nx,obj.Ny,obj.Nz]);
        
            % Create Ncoils copies of the image
            img = repmat(img, [1,1,1,obj.Ncoils]);
        
            % sensitivity maps
            img_coils = obj.smaps .* img;
        
            % Fourier transform
            data = fftshift(img_coils);
            for dim = 1:3
                data = fft(data,size(data,dim),dim);
            end
            k_coils = ifftshift(data);
        
            % undersampling mask
            k_coils = obj.sample_mask .* k_coils;
            
            % !! Commented out as it's redundant
            % % collapse dimensions one at a time
            % %k = reshape(k_coils,[M*N, Ncoils]);
            % k_vec = k_coils(:);
            % 
            % % Reshape k-space vector into multicoil 2D matrices
            % k_coils = reshape(k_vec, [M,N,Ncoils]);
        
            % Can't undo the undersampling so do nothing here
        
            % IFFT to get back multicoil images
            data = fftshift(k_coils);
            for dim = 1:3
                data = ifft(data,size(data,dim),dim);
            end
            img_coils = ifftshift(data);
        
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
            img = sum(img,4);
        
            % Vectorize
            img_vec = img(:);
        end
    end
end