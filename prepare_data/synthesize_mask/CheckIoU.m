function flag= CheckIoU(img, gt, IoU_th)
%IOU
interimg=bitand(img,gt);
unionimg=bitor(img,gt);
interimg=logical(interimg);
unionimg=logical(unionimg);
iou=sum(sum(interimg))/sum(sum(unionimg));

if iou > IoU_th
    flag = true;
else
    flag = false;
end

end

