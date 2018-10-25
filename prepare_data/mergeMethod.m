clear
clc

met_fol_rp = '/disk5/yangle/DAVIS/result/RankNet_googleF/';
res_met_rp = '/disk5/yangle/DAVIS/result/RankNet_Merge/';

met_set = dir(met_fol_rp);
met_set = met_set(3:end);

for imet = 1:length(met_set)
    met_name = met_set(imet).name;
    disp(met_name);
    claRp = [met_fol_rp, met_name, '/'];
    resRp = [res_met_rp, met_name, '/'];
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
        parfor iimg = 1:length(imgSet)
            imgName = imgSet(iimg).name;
            copyfile([imgRp, imgName], [resRp,claName,imgName]);  
        end    
    end

end
