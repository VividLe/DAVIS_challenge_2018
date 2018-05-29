% This is the reference implementation of the data augmentation described
% in the paper:
% 
%   Learning Video Object Segmentation from Static Images 
%   A. Khoreva, F. Perazzi,  R. Benenson, B. Schiele and A. Sorkine-Hornung
%   arXiv preprint arXiv:1612.02646, 2016. 
% 
% Please refer to the publication above if you use this software. For an
% up-to-date version go to:
% 
%            http://www.mpi-inf.mpg.de/masktrack
% 
% 
% THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY EXPRESSED OR IMPLIED WARRANTIES
% OF ANY KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THIS SOFTWARE.

function augment_image_and_mask(gt0,folder_path,filename)

shift=[-0.1:0.01:0.1];
IoU_th = 0.4;

% im_path=[folder_path,'/images/']; % path to image folder
bbs_path= folder_path; % path to generated mask folder
% gt_path=[folder_path,'/gt/']; % path to groundtruth maksk folder

mkdir(bbs_path); 

gt0=uint8(gt0>0);

resize=1;
im_dim1 = size(gt0,1);
im_dim2 = size(gt0,2);
num_angles = 2;
angle_step = 2;
angles = angle_step:angle_step:num_angles*angle_step;
angles = [angles, -angles];

    %contain randomicity, maybe emerge much worse mask
    count = 1;
    attempt = 0;
    while count < 5 && attempt < 10
%     for jit=1:5 
         gt1=double(gt0>0);
%          file_name = num2str((str2double(file_name)), '%05d');
         file_name=[filename '_' num2str(count)];
        
         seg=gt1;
        [M,N]=find(gt1>0);
        topM=min(M);
        bottomM=max(M);
        leftN=min(N);
        rightN=max(N);
        w=rightN-leftN;
        h=bottomM-topM; 

        se = strel('disk',1);       
        bound=imdilate(seg,se)-seg;
        [x,y]=find(bound);
        if ~isempty(x)
            if numel(x)>4
                num_points=5;
                rand_p=randsample(numel(x),num_points);
                movingPoints=zeros(num_points,2);
                fixedPoints=zeros(num_points,2);

                % translation (10% shift)
                for l=1:numel(rand_p)
                    fixedPoints(l,1)=x(rand_p(l))+ h*shift(randsample(numel(shift),1));
                    fixedPoints(l,2)=y(rand_p(l))+ w*shift(randsample(numel(shift),1));
                    movingPoints(l,1)=x(rand_p(l));
                    movingPoints(l,2)=y(rand_p(l));
                end
                % Thin-plate smoothing spline
                st = tpaps(movingPoints',fixedPoints');
                [x,y]=find(seg);
                xy=[x,y];
                avals = fnval(st,xy');
                seg1=zeros(size(seg));
                for k=1:numel(avals)
                    try
                        seg1(min(max(1,floor(avals(1,k))),size(seg,1)),min(max(1,floor(avals(2,k))),size(seg,2)))=1;        
                        seg1(min(max(1,ceil(avals(1,k))),size(seg,1)),min(max(1,ceil(avals(2,k))),size(seg,2)))=1;
                        seg1(min(max(1,floor(avals(1,k))),size(seg,1)),min(max(1,ceil(avals(2,k))),size(seg,2)))=1;
                        seg1(min(max(1,ceil(avals(1,k))),size(seg,1)),min(max(1,floor(avals(2,k))),size(seg,2)))=1;
                    end
                end

                % the mask is coarsened using dilation operation with 5 pixel radius
                se = strel('disk',5);
                seg_new = imdilate(seg1,se);
                  seg1=uint8(seg_new>0);
                else
                  seg1=uint8(seg>0);
            end
        else
          seg1=uint8(seg>0);
        end

        % skip severely deformed mask
        gt1=uint8(gt1>0);
        seg1=uint8(seg1>0);
        if CheckIoU(seg1, gt1, IoU_th)
            count = count + 1;
        else
            attempt = attempt + 1;
            disp('skip severely deformed mask');
            continue;
        end
        
        str_ang = '_0';
        seg_write = seg1 * 255;
        imwrite(seg_write, [bbs_path file_name str_ang  '.png']);

        for a = 1:length(angles)
                angle = angles(a);
                try
                seg1_rot_crop = rotate_image(seg1, angle, 'nearest');
                 catch
%                     im1_rot_crop =0; 
                end

                if resize
                    seg1_rot_crop = imresize(seg1_rot_crop, [im_dim1, im_dim2], 'nearest');
                end

                seg1_rot_crop=uint8(seg1_rot_crop>0);
                
                if angle < 0 
                    str_ang = int2str(angle);
                else
                    str_ang = ['+', int2str(angle)];
                end
                seg1_write = seg1_rot_crop * 255;
                imwrite(seg1_write, [bbs_path file_name str_ang '.png']);
        end

     end
end
