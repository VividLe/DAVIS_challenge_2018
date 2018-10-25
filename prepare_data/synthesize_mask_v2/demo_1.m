clear, clc

cla_folder_path = '/disk5/yangle/DAVIS/dataset/synthe_mask/';
img_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/mImages/';
gt_rp = '/disk5/yangle/DAVIS/dataset/sep_mask/';

img_set = dir([img_rp, '*.png']);
parfor iimg = 1:6000
% for iimg = 1:1
    disp(iimg);
    img_name = img_set(iimg).name;
    file_name = img_name(1:end-4);
    mask_set = dir([gt_rp, file_name, '*']);
    
    for imas = 1:length(mask_set)
        mask_name = mask_set(imas).name;
        mask = imread([gt_rp, mask_name]);
        if max(max(mask)) == 0
           continue; 
        end
        
        gt_file_name = mask_name(1:end-4);
        
        img = imread([img_rp, img_name]);
        mask = imread([gt_rp, mask_name]);
        folder_path = [cla_folder_path, gt_file_name, '/'];

%         if exist(folder_path, 'dir')
%             continue;
%         end

        % deform segmentation masks
        try
            augment_image_and_mask(mask, folder_path, gt_file_name);
        catch
        end
        
    end
end
