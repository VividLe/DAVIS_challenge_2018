clear
clc

claRp = '/disk5/yangle/DAVIS/result/mask/test-dev/';
resRp = '/disk5/yangle/DAVIS/result/mask/test-devM/';
if ~exist(resRp, 'dir')
    mkdir(resRp);
end

claSet = dir(claRp);
claSet = claSet(3:end);
for icla=1:length(claSet)
    claName = claSet(icla).name;
    disp(claName);
    imgRp = [claRp,claName,'/'];
    imgSet = dir([imgRp, '*.png']);
    for iimg = 1:length(imgSet)
        imgName = imgSet(iimg).name;
        copyfile([imgRp, imgName], [resRp,claName,imgName]);
%         img = imread([imgRp, imgName]);
%         imgNewName = [claName, imgName(1:end-4), '.png'];
%         imwrite(img, [resRp,imgNewName], 'png');    
    end    
end

% claSet = dir(claRp);
% claSet = claSet(3:end);
% for icla=1:length(claSet)
%     claName = claSet(icla).name;
%     imgRp = [claRp,claName,'/'];
%     imgSet = dir([imgRp, '*.jpg']);
%     parfor iimg = 1:length(imgSet)
%         imgName = imgSet(iimg).name;
%         imgNewName = [claName,imgName];
%         copyfile([imgRp, imgName], [resRp,imgNewName]);        
%     end    
% end
