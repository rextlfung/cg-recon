%% Compare ground truth and recon'd images
function compareImages(img_gt, img_recon)
    tiledlayout('flow','TileSpacing','tight');
    nexttile; im(abs(img_gt)); title('Ground Truth magnitude'); colorbar;
    nexttile; im(angle(img_gt)); title('Ground Truth phase'); colorbar;
    nexttile; im(abs(img_recon)); title('Recon magnitude'); colorbar;
    nexttile; im(angle(img_recon)); title('Recon phase'); colorbar;
    nexttile; im(abs(img_gt - img_recon)); title('Magnitude of diff img'); colorbar;
    nexttile; im(angle(img_gt - img_recon)); title('Phase of diff img'); colorbar;
end