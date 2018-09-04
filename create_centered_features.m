    % 
  %  load()
% clear all;
% 
% 
% % load('tensor_bands_india.mat');
% % im_cluster=imread('india_GMM_cl5.tif');
% %im_cluster=
% 
% 
% % load('tensor_bands_aviris.mat');
% % im_cluster=imread('aviris_GMM_cl6.tif');
% 
% load('tensor_bands.mat');
% 
% im_cluster=imread('out_KM_cl7.tif');
% 
% [bands,rows,cols]=size(T);
% % im_cluster=im_cluster(1:rows,1:cols);
% %[r,c,bands]=size(im);
% class_num=max(max(im_cluster));
% Ttemp=T;
% bsize=16;

% clear all;
% clear all;
% load('tensor_bands.mat');
%        
% % % %load('tensor_int32cl1.mat');
% 
% T(5:10 ,:,:)=[];
% Ttemp=T;  
% 
% oo=0;
% [bands row_original col_original]=size(Ttemp);
    

bbegin=1;
ccount=1;
 begin=1;
ffcount=1;
% clear divide_val;
 sbegin=1;
%  im_mask=im_cluster;
%    im_mask=im_mask(1:rows,1:cols);
% for bsize=16:8:16
%     clear patches;
%     pcount=1;
%     bbegin=1;
%       ccount=1;
% for c=1:7
%     
%     %%%aviris
%     
% %   if  c==4 || c==5 ||c==3 ||c==6
% %        continue
% %    end
%     %%%india
% %    if c==2 || c==4 || c==1||c==5
% %        continue
% %    end
% 
% %%%nias
% % if c==2 || c==6 || c==1||c==5
% %        continue
% % end
% 
%     c
%   if c==2
%         continue
%     end
%   % im_mask=padarray(im_mask,[bsize/2 bsize/2],'both');
%    inds_not=find(im_mask~=c);
%    inds_mask=find(im_mask==c);
%    
%    [inds_row,inds_col]=ind2sub(size(im_mask),find(im_mask==c));
%     im_mask(inds_not)=0;
%     im_mask(inds_mask)=1;
%     im_mask2=imdilate(im_mask,ones(bsize/2,bsize/2));
% % imshow(im_mask*50)
% % hhh
%     
%     rows=row_original;
%     cols=col_original;
%     bsize=16;
%     pcount=1;
%    % sind=size(inds_row);
%     rand_num=5000;
%     
%     for r=1:rand_num%sind(1)\
%         r
%         i=randi([bsize/2,rows-bsize/2]);
%         j=randi([bsize/2,cols-bsize/2]);
%         for b=1:bands
%            temp=Ttemp(b,:,:);
%            temp=reshape(temp(1,:),[rows cols]);
% %            temp=temp-10;
% %            temp=temp/max(max(temp));
%          %  temp=padarray(temp,[bsize/2 bsize/2],'both');
%         
%            patch=temp(i-bsize/2+1:i+bsize/2,j-bsize/2+1:j+bsize/2);
%           % imshow((patch-min(min(patch))/(max(max(patch))-min(min(patch)))));
%            patches(b,pcount,:)=reshape(patch,[1 bsize*bsize]);
%            
%         end
%         pcount=pcount+1;
%     end
%    hhh
% 
% %     comp_num=7;
% %     Uhat=cpd(double(patches),comp_num);
% %     Vec_block=Uhat{3};
% %     Vec_patch=Uhat{2};
% %     Vec_band=Uhat{1};
% %     sort_blocks;
% %     shbegin=1;
% %    for sh=0:0
% %    get_patch_tensor;
% %     ccount=1;
% %     bbegin=1;
% % 
% %    end
%   %   break
%     % end
%  %end
% 
% % xxxx=2
%     comp_num=3  %bsizcoe^2/4;
%     sb=1;
%     num=1;
%     
%      for i=1:num
%    % Uhat=cpd(double(patches),comp_num);
%     tensor_struct;
%     sort_blocks;
%     if sb==1
%         blocks_sort=Vec_block_sort;
%         band_sort=Vec_band_sort;
%         sb=0;
%     else
%        blocks_sort=blocks_sort+ Vec_block_sort;
%        band_sort=band_sort+Vec_band_sort;
%     end
%     
%     end
%    blocks_sort=blocks_sort/num;
%    band_sort=band_sort/num;
%    Vec_band=band_sort;
%    Vec_block_sort=blocks_sort;
bsize=yblock;
    
  sbegin=1;
  %clear total_sh;
clear total_comp;
clear patch_factor;
sh=0;
% 
%      begin=1;
%    get_patch_tensor;
% 
% ffcount=1;
%     solve_factors;
%     norms=sqrt(sum(patch_factors.^2, 1));
%     dd=([0,diff(norms,1)]).^2;
%     dd=dd./norms;
%    
%     bbegin=1;
%     ccount=1;
% get_comps;
% 
% 
% %save('india_shift_class1_block16.mat','total_comp');
% save('nias_shift_rat_class1_block16.mat','total_comp');

%load('tensor_DC.mat');
  %load('tensor_aviris.mat');
%load('tensor_bands1.mat');
%T(5:10 ,:,:)=[];
load('/Users/Golnoosh/Desktop/SAIE_lab/Pavia/PaviaU.mat');
block_s=32;
[r,c,bands]=size(paviaU);
%for i=1:bands
    paviaU=paviaU(1:floor(r/block_s)*block_s,1:floor(c/block_s)*block_s,:);%paviaU(1:floor(r/block_s)*block_s,1:floor(c/block_s)*block_s,i);
%end
%%jj
T=permute(paviaU,[3 1 2]);;;
%load('tensor_DC.mat');
%load('tensor_aviris.mat');
%load('tensor_bands1.mat');
% load('tensor_indiawww.mat');      
% % %load('tensor_int32cl1.mat');

%T(5:10 ,:,:)=[];


Ttemp=double(T); 
Ttemp=T;
[bands rows cols]=size(Ttemp);
      begin=1;
   
   get_patch_tensor;
     

ffcount=1; 
    solve_factors;
    norms=sqrt(sum(patch_factors.^2, 1));
    [sortedValues,sortIndex] = sort(A(:),'descend');
    [norms,sortIndex] = sort(norms,2,'descend');   % No UNIQUE here
    inds =sortIndex(1:comp_num);
    patch_factors=patch_factors(:,inds);
    bbegin=1;
    ccount=1;
get_comps;
%save('nias_shift_rat_class1_block16_1.mat','total_comp');


% save('india_block_www.mat','total_comp');
% hhhhhh
% 
%   load('tensor_bands0.mat');
% 
% T(5:10 ,:,:)=[];
% Ttemp=T;
% [bands rows cols]=size(Ttemp);
%       begin=1;
%    get_patch_tensor;
% 
% ffcount=1;
%     solve_factors;
%   norms=sqrt(sum(patch_factors.^2, 1));
%     [sortedValues,sortIndex] = sort(A(:),'descend');
%     [norms,sortIndex] = sort(norms,2,'descend');   % No UNIQUE here
%     inds =sortIndex(1:5);
%     patch_factors=patch_factors(:,inds);
%     bbegin=1;
%     ccount=1;
% get_comps;
% 
% 
% %save('india_shift_class1_block16.mat','total_comp');
% save('nias_shift_rat_class1_block16_0.mat','total_comp');
% 
%  
% 
%  load('tensor_bands2.mat');
% 
% T(5:10 ,:,:)=[];
% Ttemp=T;
% [bands rows cols]=size(Ttemp);
%       begin=1;
%    get_patch_tensor;
% 
% ffcount=1;
%     solve_factors;
%  norms=sqrt(sum(patch_factors.^2, 1));
%     [sortedValues,sortIndex] = sort(A(:),'descend');
%     [norms,sortIndex] = sort(norms,2,'descend');   % No UNIQUE here
%     inds =sortIndex(1:6);
%     patch_factors=patch_factors(:,inds);
%      bbegin=1;
%     ccount=1;
% get_comps;
% 
% 
% %save('india_shift_class1_block16.mat','total_comp');
% %save('nias_shift_rat_class1_block16_2.mat','total_comp');
% % %save('aviris_shift_class1_block16.mat','total_comp'); 
%  %   hhh
%     
% 
%     
% %        im_mask2=double(im_mask2);
% %     for b=1:bands
% %        temp=Ttemp(b,:,:);
% %         temp=reshape(temp(1,:),[rows cols]);
% %         temp=temp.*(im_mask2);%+200*(ones(size(im_mask2))-im_mask2); 
% %        % imtool(temp/2);
% %        Ttemp(b,:,:)=temp;
% %     end
% %   
% %     [comps,Vec2]=tensor_shift(Ttemp,comp_num);
% %     [rr,cc,comps_n]=size(comps);
% %     for c=1:comps_n
% %        cc=comps(:,:,c);
% %        cc=cc.*double(im_mask);
% %        Ttemp(bands+c,:,:)=cc;
% %     end
% %     save('int_components_16nosh_6c_32cl1.mat','Ttemp');
% %     ggg
% % %end