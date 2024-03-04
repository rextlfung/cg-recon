%% Compare ground truth and recon'd images
function compareImages(img_gt, img_us, img_recon)
    tiledlayout('flow','TileSpacing','tight');
    nexttile; im(abs(img_gt)); title('Ground Truth magnitude'); colorbar;
    nexttile; im(abs(img_us)); title('Undersampled magnitude'); colorbar;
    nexttile; im(abs(img_recon)); title('Recon magnitude'); colorbar;

    % Differences
    img_diff = img_gt - img_recon;
    nexttile; im(abs(img_diff)); title('Magnitude of diff img'); colorbar;
    MSE = mean(img_diff.^2,'all')
end