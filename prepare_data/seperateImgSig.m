clear
clc

imgOriRp = '/disk5/yangle/DAVIS/dataset/RankingTest/sep_object/';
claResRp = '/disk5/yangle/DAVIS/dataset/RankingTest/sep_object_sep/';

if ~exist(claResRp, 'dir')
    mkdir(claResRp);
end

imgSet = dir([imgOriRp, '*.png']);
for iimg = 1:length(imgSet)
    imgName = imgSet(iimg).name;    
    claName = imgName(1:end-9);
%     claName = imgName(1:end-10);
    imgResRp = [claResRp, claName, '/'];
    if ~exist(imgResRp, 'dir')
        mkdir(imgResRp);
    end
    resImgName = imgName(end-8:end);
%     resImgName = imgName(end-9:end);
    copyfile([imgOriRp, imgName], [imgResRp, resImgName]);    
end


