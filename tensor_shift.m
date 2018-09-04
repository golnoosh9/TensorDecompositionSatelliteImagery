% % % %%%%%get_blocks
%%%data format: band_num,rows,cols
clear all;

load('/Users/Golnoosh/Desktop/SAIE_lab/Pavia/PaviaU.mat');
load('/Users/Golnoosh/Desktop/SAIE_lab/Pavia/PaviaU_gt.mat');
hyper=1;
block_s=32;
[r,c,bands]=size(paviaU);
%for i=1:bands
    paviaU=paviaU(1:floor(r/block_s)*block_s,1:floor(c/block_s)*block_s,:);%paviaU(1:floor(r/block_s)*block_s,1:floor(c/block_s)*block_s,i);
%end
%%jj
T=permute(paviaU,[3 1 2]);;
% %load('tensor_DC.mat');
% %load('tensor_aviris.mat');
% 
% %load('tensor_bands1.mat');
% % load('tensor_indiawww.mat');      
% % % %load('tensor_int32cl1.mat');
% 
%T(5:10 ,:,:)=[];


Ttemp=double(T); 

   oo=0;
[bands row_original col_original]=size(Ttemp);


bsize=8
begin=1
bbegin=1
count=1
bcount=1

%comps=bands+5;%%%10 components

%for i=1:compsc
shbegin=1;
for ysize=1:1 
    
    ysize
   xsize=ysize;
%while xsize>1
    begin=1;
    sbegin=1;
    yblock=power(2,ysize)
    xblock=power(2,xsize) 
    comp_num=30;
    comps=comp_num;
    
    %-xblock:ceil(xblock/4):xblock%min(3,block/2)
%     for i=1:bands%%%go over all bands
%         sbegin=1;
%         for ang=0:0
%             ang;
%           
%          
%     
%              
%        nim=reshape(Ttemp(i,:,:),[row_original col_original]);
% %        imtool(nim);
% %        hh
%       nim=circshift(nim,[s s]);
%       nim=imrotate(nim,ang);
%       
%      im=nim(1:floor(row_original/xblock)*xblock,1:floor(col_original/yblock)*yblock)*comps;
%   
%  
%      [row col]=size(im);
%    
%      tiled=mat2tiles(im,xblock,yblock);
%      temp_image=cell2mat(tiled);
% 
%      new_image = reshape(cellfun(@(x)reshape(x,[1 xblock*yblock ]),tiled,'UniformOutput',false),[row*col/(xblock*yblock) 1]);
%      
% %      if sbegin==1
% %          shift_image=cell2mat(new_image);
% %         
% %          sbegin=0;
% %      else
% %          shift_image=[shift_image;cell2mat(new_image)];%%%concatenate shifted images to pixel dimension
% %      end
%      
% %     
% %          
% %      if begin==1
% %          tens_image=cell2mat(new_image);
% %           begin=0;
% %       else
% %           tens_image(:,:,i)=cell2mat(new_image);
% %         
% %      end
%       
%          
%               if begin==1
%          tens_image=cell2mat(new_image);
%           begin=0;
%       else
%           tens_image(:,:,bcount)=cell2mat(new_image);
%         
%               end
%               bcount=bcount+1;
%    
%     end
%         end
 
    
 %   patches=tens_image;
    rows=row_original;
    cols=col_original;
    paviaU_gt=paviaU_gt(1:rows,1:cols);
    bsize=power(2,ysize);
    pcount=1;
   % sind=size(inds_row);
    rand_num=500;
    clear patches;
    for r=1:rand_num%sind(1)\
        r
        i=randi([bsize/2+1,rows-bsize/2]);
        j=randi([bsize/2+1,cols-bsize/2]);
        for b=1:bands
           temp=Ttemp(b,:,:);
           
           temp=reshape(temp(1,:),[rows cols]);
%            temp=temp-10;
%            temp=temp/max(max(temp));
         %  temp=padarray(temp,[bsize/2 bsize/2],'both');
        
           patch=temp(i-bsize/2+1:i+bsize/2,j-bsize/2+1:j+bsize/2);
          % imshow((patch-min(min(patch))/(max(max(patch))-min(min(patch)))));
           patches(b,pcount,:)=reshape(patch,[1 bsize*bsize]);
           
        end
        pcount=pcount+1;
    end

    sb=1;
    num=1;
    
     for i=1:num
   % Uhat=cpd(double(patches),comp_num);
    tensor_struct;
    sort_blocks;
    if sb==1
        blocks_sort=Vec_block_sort;
        band_sort=Vec_band_sort;
        sb=0;
    else
       blocks_sort=blocks_sort+ Vec_block_sort;
       band_sort=band_sort+Vec_band_sort;
    end
    
    end
   blocks_sort=blocks_sort/num;
   band_sort=band_sort/num;
   Vec_band=band_sort;
   Vec_block_sort=blocks_sort;

   

[bands,rows,cols]=size(Ttemp);
     
patch_factors=Vec_block_sort;

%bsize=power(2,ysize);
bbegin=1;
ccount=1;
%shbegin=1;
 nn=0;
for sh=0:0
   create_centered_features;
    if shbegin==1
        total_sh=total_comp;
        shbegin=0;
    else
        total_sh=(total_comp+total_sh);
    end
        
   end

% 
% nn=0;
% for c=1:comp_num
%     comp1=total_comp(:,:,c);
%     nn=nn+norm(comp1);
% end
% nn=nn/comp_num;
% bbegin=1;
% 
% ccount=1;
% for c=1:comp_num
%     comp1=total_comp(:,:,c);
%    % comp1=(comp1-min(min(comp1)))/(max(max(comp1))-min(min(comp1)));
%     norm(comp1)
%     if norm(comp1)<nn 
%         continue
%     end
%       if bbegin==1
%           
%           % block_shapes=Vec_block(:,c);
%          total_ch=(comp1-min(min(comp1)))/(max(max(comp1))-min(min(comp1)));
%          bbegin=0;
%     
%         else
%           % block_shapes(:,ccount)=Vec_block(:,c);
%             total_ch(:,:,ccount)=(comp1-min(min(comp1)))/(max(max(comp1))-min(min(comp1))) ;
%         end
%         ccount=ccount+1;
%     
end

% end

 total_comp=total_sh;
% total_comp=total_comp(:,:,15);
total_comp=reshape(total_comp,[size(total_comp,1)*size(total_comp,2) size(total_comp,3)]);
paviaU_gt=reshape(paviaU_gt,[size(paviaU_gt,1)*size(paviaU_gt,2) 1]);
%%save('india_shift_class1_block16.mat','total_comp');

train_patch=[];%patch_factorss()
train_labels=[];%
class_num=max(max(paviaU_gt));
for ii=1:class_num
    inds=find(paviaU_gt==ii);
    size_i=size(inds,1);
 inds=randi( size_i,1,floor(0.3*size_i));  


size_i=size(inds,1);
size_t=size(total_comp,3);
tc=total_comp(inds,:);


gt=paviaU_gt(inds);
gt=gt(:);
  train_patch=cat(1,train_patch,tc);%%sample image from ith class
  train_labels=cat(1,train_labels,gt);
  
  if ii==10
      jjj
  end
end

inds=find(paviaU_gt>0);
test_total=total_comp(inds,:);
label_total=paviaU_gt(inds);

Model=svm.train(train_patch,train_labels','kernel_function','rbf');
im_predict=svm.predict(Model,test_total);
%% Find Accuracy
Accuracy=mean(label_total==im_predict')*100;
fprintf('\nAccuracy =%d\n',Accuracy)
% 
% save('pavia.mat','total_comp'); 
hhh



    
    
    
