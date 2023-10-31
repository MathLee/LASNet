clear; close all; clc;
%Path of semantic gts
gtPath = '/Volumes/RGBT_Semantic_Seg/PST900_RGBT_Dataset/labels/';

savePath = '/Volumes/RGBT_Semantic_Seg/PST900_RGBT_Dataset/binary_labels/';

gts = dir([gtPath '*.png']);
gtsNum = length(gts);

               
for i=1:gtsNum
   gt_name = gts(i).name();
   
   gt = imread(fullfile(gtPath, gt_name));
   
   gt(find(gt>1)) = 255;
  
   imwrite(gt, [savePath gt_name] );
                
end

