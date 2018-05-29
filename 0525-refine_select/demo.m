addpath(genpath('./'))

test_root_path = '/disk5/yangle/DAVIS/dataset/RankingTest/process_refine/';

threshold = 0.01;
maxIterations = 5;

comask = imread([test_root_path, 'comask.png']);
salmap = imread([test_root_path, 'salmap.png']);
img = imread([test_root_path, 'img.jpg']);

box = getBbox(comask);
factor=0.2;
[x_min,x_max,y_min,y_max] = enlarge_box(box, factor);
[rows, cols] = size(comask);
if x_min < 1
    x_min = 1;
end
if x_max > rows
    x_max = rows;
end
if y_min < 1
    y_min = 1;
end
if y_max > cols
   y_max = cols; 
end

% saliency map
salmap = rgb2gray(salmap);
salmap = imresize(salmap, size(comask), 'bilinear');
% set zero
salmap(1:x_min,:) = 0;
salmap(x_max:rows, :) = 0;
salmap(:,1:y_min) = 0;
salmap(:,y_max:cols) = 0;

% noisy map
noise_point = 0.1*rand([rows, cols]);
noise_point(1:x_min,:) = 0;
noise_point(x_max:rows, :) = 0;
noise_point(:,1:y_min) = 0;
noise_point(:,y_max:cols) = 0;

saliency_map = 0.8 * double(salmap) / 255;
GausKernl = fspecial('gaussian', [50, 50], 10);
unary_map = imfilter(saliency_map,GausKernl,'same');
potential_map = noise_point + unary_map;
potential_map(potential_map > 1) = 1;

seg = st_segment(img,potential_map,threshold,maxIterations);
seg = uint8(255*double(seg));
imwrite(seg,[test_root_path, 'refine.png'],'png');
