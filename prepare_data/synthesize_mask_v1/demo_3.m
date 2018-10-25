clear, clc

cla_folder_path = '/disk5/yangle/DAVIS/dataset/saliency/train_syn_formask/';
img_rp = '/disk5/yangle/DAVIS/dataset/saliency/train/';
gt_rp = '/disk5/yangle/DAVIS/dataset/saliency/trainannot/';

img_set = dir([img_rp, '*.png']);
parfor iimg = 1:8000
%for iimg = 1:2
    disp(iimg);
    img_name = img_set(iimg).name;
    file_name = img_name(1:end-4);
    % file_name = num2str((str2double(filename)), '%06d');
    img = imread([img_rp, img_name]);
    gt = imread([gt_rp, img_name]);
    folder_path = [cla_folder_path, file_name, '/'];
    
    if exist(folder_path, 'dir')
        continue;
    end
    
    % deform segmentation masks
    try
        augment_image_and_mask(gt, folder_path,file_name);
    catch
    end
end
