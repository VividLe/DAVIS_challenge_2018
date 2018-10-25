clear, clc

DISCARD_TH = 500;

mask_rp = '/disk5/yangle/DAVIS/result/mask/maskrcnn_Caffe2/SegMask/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/';
cate_res_rp = '/disk5/yangle/DAVIS/result/RankNet/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/';

mask_set = dir([mask_rp, '*.png']);
parfor igt = 1:length(mask_set)
    mask_name = mask_set(igt).name;
    cate_name = mask_name(1:end-9);
    img_base_name = mask_name(end-8:end-4);
    img_res_rp = [cate_res_rp, cate_name, '/', img_base_name, '/'];
    if ~exist(img_res_rp, 'dir')
        mkdir(img_res_rp);
    end
    
    mask_ori = imread([mask_rp, mask_name]);
    [rows, cols] = size(mask_ori);
    
    obj_num = max(max(mask_ori));
    for iobj = 1:obj_num
        mask = uint8(zeros(rows, cols));
        % discard subtle object
        [x,~] = find(mask_ori == iobj);
        if length(x) < DISCARD_TH
            continue
        end
        mask(mask_ori == iobj) = 255;
        ord_str = num2str(iobj, '%02d');
        save_file = [img_res_rp, ord_str, '.png'];
        imwrite(mask, save_file, 'png');
    end

end