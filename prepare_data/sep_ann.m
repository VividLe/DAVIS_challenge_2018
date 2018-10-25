clear, clc

gt_rp = '/disk5/yangle/DAVIS/result/mask/maskrcnn_Caffe2/SegMask_vis/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/';
sep_rp = '/disk5/yangle/DAVIS/result/RankNet/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/';

gt_set = dir([gt_rp, '*.png']);
%for igt = 1:5
for igt = 1:length(gt_set)
    gt_name = gt_set(igt).name;
	disp(gt_name);
    gt = imread([gt_rp, gt_name]);
    gt(gt == 255) = 0;
    [rows, cols] = size(gt);

    base_name = gt_name(1:end-4);
    obj_num = max(max(gt));
    for iobj = 1:obj_num
        mask = uint8(zeros(rows, cols));
        mask(gt == iobj) = 255;
        ord_str = num2str(iobj, '%02d');
        save_file = [sep_rp, base_name, ord_str, '.png'];
        imwrite(mask, save_file, 'png');
    end

end