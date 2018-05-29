clear
clc

imgOriRp = '/disk5/yangle/DAVIS/dataset/RankingTest/sep_object/';
claResRp = '/disk5/yangle/DAVIS/dataset/RankingTest/sep_object_sep/';

if ~exist(claResRp, 'dir')
    mkdir(claResRp);
end

imgSet = dir([imgOriRp, '*.png']);
for iimg = 1:length(imgSet)
    img_name = imgSet(iimg).name;    
    claName = img_name(1:end-11);
    img_base_name = img_name(end-10:end-6);
    imgResRp = [claResRp, claName, '/', img_base_name, '/'];
    if ~exist(imgResRp, 'dir')
        mkdir(imgResRp);
    end
    resImgName = img_name(end-5:end);
    copyfile([imgOriRp, img_name], [imgResRp, resImgName]);    
end


