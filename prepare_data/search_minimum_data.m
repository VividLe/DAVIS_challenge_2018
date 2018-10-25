clear, clc

MIN_X = 300;

gt_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/MergeFiles/DAVIS/Annotations/mAnnotations/';
sep_rp = '/disk5/yangle/DAVIS/result/RankNet/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/';

gt_set = dir([gt_rp, '*.png']);
%for igt = 1:5
for igt = 1:length(gt_set)
    gt_name = gt_set(igt).name;
	% disp(gt_name);
    gt = imread([gt_rp, gt_name]);
    gt(gt == 255) = 0;
%     [rows, cols] = size(gt);

    base_name = gt_name(1:end-4);
    obj_num = max(max(gt));
    for iobj = 1:obj_num
        [x,~] = find(gt == iobj);
        if (~isempty(x)) && (length(x) < MIN_X)
%             MIN_X = length(x);
            fprintf('image: %s, the %d object, contains %d pixels\r\n', gt_name, iobj, length(x));
        end
%         mask = uint8(zeros(rows, cols));
%         mask(gt == iobj) = 255;
%         ord_str = num2str(iobj, '%02d');
%         save_file = [sep_rp, base_name, ord_str, '.png'];
%         imwrite(mask, save_file, 'png');
    end

end