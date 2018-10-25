clear, clc

DISCARD_TH = 500;

mask_rp = '/disk5/zhangdong/database/VOC2012_SEG_AUG/newsegmentations/';
cate_res_rp = '/disk5/yangle/DAVIS/dataset/Objectness/deform_mask/';

mask_set = dir([mask_rp, '*.png']);
parfor igt = 1:length(mask_set)
    mask_name = mask_set(igt).name;
    disp(mask_name);
    mask_ori = imread([mask_rp, mask_name]);
    [rows, cols] = size(mask_ori);
    
    obj_num = max(max(mask_ori));
    for iobj = 1:obj_num
        mask = uint8(zeros(rows, cols));
        mask(mask_ori == iobj) = 255;
        ord_str = num2str(iobj);
        save_file = [cate_res_rp, mask_name(1:end-4), '_', ord_str, '.png'];
        imwrite(mask, save_file, 'png');
    end

end