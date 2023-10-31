clear; close all; clc;
%Path of semantic gts
ssgtPath = '/Volumes//RGBT_Semantic_Seg/PST900_RGBT_Dataset/labels/';

savePath = '/Volumes/RGBT_Semantic_Seg/PST900_RGBT_Dataset/bound/';

ssgts = dir([ssgtPath '*.png']);
gtsNum = length(ssgts);

               
for i=1:gtsNum
   ssgt_name = ssgts(i).name();
   
   ssgt = imread(fullfile(ssgtPath, ssgt_name));
   
   [h,w] = size(ssgt);
   
   bound = zeros(size(ssgt));
   
   padmap = zeros(h+4, w+4);
   
   padmap(3:h+2,3:w+2) = ssgt;
   
   
   for hh = 1:h
       for ww = 1:w
           slidewindow = padmap(hh:hh+4, ww:ww+4);
           class = unique(slidewindow);
           if length(class)>=2
               bound(hh,ww) = 255;
           end
       end
   end
   
  
   imwrite(uint8(bound), [savePath ssgt_name] );
                
end

