"""
Classify a multi-band, satellite image.
Usage:
    classify.py <input_fname> <landsat_data> <cloud_cover> <train_data_path> <output_fname> <class_number> <class_number_correct> [--method=<classification_method>]
                                                               [--validation=<validation_data_path>]
                                                               [--verbose]
    classify.py -h | --help
The <input_fname> argument must be the path to a GeoTIFF image.
The <train_data_path> argument must be a path to a directory with vector data files
(in shapefile format). These vectors must specify the target class of the training pixels. One file
per class. The base filename (without extension) is taken as class name.
If a <validation_data_path> is given, then the validation vector files must correspond by name with
the training data. That is, if there is a training file train_data_path/A.shp then the corresponding
validation_data_path/A.shp is expected.
The <output_fname> argument must be a filename where the classification will be saved (GeoTIFF format).
No geographic transformation is performed on the data. The raster and vector data geographic
parameters must match.
Options:
  -h --help  Show this screen.
  --method=<classification_method>      Classification method to use: random-forest (for random
                                        forest) or svm (for support vector machines)
                                        [default: random-forest]
  --validation=<validation_data_path>   If given, it must be a path to a directory with vector data
                                        files (in shapefile format). These vectors must specify the
                                        target class of the validation pixels. A classification
                                        accuracy report is writen to stdout.
  --verbose                             If given, debug output is writen to stdout.
"""
#import spams
import math
import random
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from docopt import docopt
from osgeo import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import cv2
import scipy.io as sio
from sklearn import manifold, datasets
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import manifold, datasets
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from  scipy import ndimage
import math
import skfuzzy as fuzz
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sktensor import dtensor, cp_als
import tensorflow as tf



logger = logging.getLogger(__name__)
padd=1
cloud_nums=[5,4,3]
# A list of "random" colorsAgri, Bare, BareAgri, Built, Forest, Water
COLORS = [
          "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
          "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
          "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
          "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
          "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
          "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
          "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
          "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
          "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
          "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
          "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
          "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
          "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
          "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
          "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
          "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
          ]

def convertToOneHot(vector, num_classes=None):
    """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.
        
        Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v
        
        [[0 1 0 0 0]
        [1 0 0 0 0]
        [0 0 0 0 1]]
        """
    
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    
    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)
    
    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, projection, target_value=1,
                            output_fname='', dataset_format='MEM'):
    """
    Rasterize the given vector (wrapper for gdal.RasterizeLayer). Return a gdal.Dataset.
    :param vector_data_path: Path to a shapefile
    :param cols: Number of columns of the result
    :param rows: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    :param target_value: Pixel value for the pixels. Must be a valid gdal.GDT_UInt16 value.
    :param output_fname: If the dataset_format is GeoTIFF, this is the output file name
    :param dataset_format: The gdal.Dataset driver name. [default: MEM]
    """
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    print vector_data_path
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    featureCount = layer.GetFeatureCount()
    print featureCount
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
 
    return target_ds

def create_stacked_raster(raster,bands_data):
    height, width = raster.shape
    
    raster=np.lib.pad(raster, ((padd, padd), (padd, padd)), 'edge')
    #    filter_blurred_f = ndimage.gaussian_filter(raster, 1)
    #    alpha = 1
    #raster = raster + alpha * (raster - filter_blurred_f)
    ####loop over offsets
    
    for i in range(2*padd+1):
        for j in range(2*padd+1):
            
            shift_raster=raster[i:i+height,j:j+width]
            
            bands_data.append(shift_raster)
    
    return bands_data

def get_DCT_pix(raster,padd,coeff_num):
    total=[]
    rastert=np.zeros((raster.shape[0],raster.shape[1],coeff_num**2),dtype=raster.dtype)
    print rastert.shape
    raster=np.lib.pad(raster, ((padd/2, padd/2), (padd/2, padd/2)), 'edge')
    for i in range(rastert.shape[0]):
        for j in range(rastert.shape[1]):
            coeffs=cv2.dct(raster[i:i+padd,j:j+padd])
            count=0
            for c1 in range(0,coeff_num):
                for c2 in range(0,coeff_num):
                    rastert[i,j,count]=coeffs[c1,c2]
                    count+=1
    return rastert


def get_coeff_pix(raster,padd,bands):
    total=[]
    padd=int(padd/2)
    rastert1=np.zeros((raster.shape[0],raster.shape[1]),dtype=raster.dtype)
    rastert2=np.zeros((raster.shape[0],raster.shape[1]),dtype=raster.dtype)
    raster=np.lib.pad(raster, ((padd, padd), (padd, padd)), 'edge')
    for i in range(rastert1.shape[0]):
        for j in range(rastert1.shape[1]):
            b1=(raster[i:i+padd,j:j+2*padd])
            b2=(raster[i+padd:i+2*padd,j:j+2*padd])
            c1=np.correlate(np.ndarray.flatten(b1),np.ndarray.flatten(b2))
            rastert1[i,j]=np.fabs(c1)
            
            b1=(raster[i:i+2*padd,j:j+padd])
            b2=(raster[i:i+2*padd,j+padd:j+2*padd])
            c2=np.correlate(np.ndarray.flatten(b1),np.ndarray.flatten(b2))
            rastert2[i,j]=np.fabs(c2)
            
    
    bands.append(rastert2)
    bands.append(rastert1)
    return bands
    
            
def get_gabor(nratio,ksize,gauss_std,sinus_wlength,moves,slopes,height,width): 
        gabors=[]
        for theta in slopes:
            for m in range(moves):
                
                tratio=nratio[m:m+height,m:m+width]
                #  print tratio.shape
                tpratio=np.lib.pad(tratio, ((ksize/2, ksize/2), (ksize/2, ksize/2)), 'edge')
                ###before theta: gaussian dtandard deviation
                ###increase parameter after theta to decrease number of lines
                kern = cv2.getGaborKernel((ksize, ksize),gauss_std, theta, sinus_wlength, 0.01, 0, ktype=cv2.CV_32F)
                kern=kern.astype(np.float32);
                # cv2.imwrite('kern'+str(ksize)+str(sinus_wlength)+'.png',kern*200)
                result=cv2.matchTemplate(tpratio,kern,2)
                
                gabors.append(result)
        gabors=np.asarray(gabors)
#        gab_var=np.var(gabors,axis=0)
#        gab_mean=np.mean(gabors,axis=0)
#        mask = np.greater(np.fabs((gab_mean)), 0)
#        #gab_var=np.choose(mask,(0,gab_var/gab_mean))
#        mask = np.greater(np.fabs(np.max(gab_var)), 0)
#        gab_var=np.choose(mask,(0,gab_var/np.max(gab_var)))
        return gabors/np.max(gabors)
 
def blockshaped(arr, nrows, ncols):
    """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size
        
        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))


      



def get_samples(raster,in_array,compute,arg_num):
    
    sample_cl=8
    cluster_sample_num=1000
    bands_all=[]
    bands_spectral=[]
    bands_texture=[]
    bands_texture2=[]
    bands_texture3=[]
    bands_texture4=[]
    bands_n=[]
    bands_short=[]
    bands=[]
    bands_d=[]
    win_rows=[25]
    # fIndex=np.ndarray.flatten(Index)
    NIRb=raster.GetRasterBand(1)
    NIR=NIRb.ReadAsArray()
    begin=1
    get_ratio=1
    neighbor=0
    degree=2
    n_comp=10
    block_size=8
    coeff_num=2
    ksize=8
    cloud=5
    accuracy=20.0
    moves=1
    band_break=4
    neighbor_num=8
    gabors=[]
    slopes=np.arange(0, np.pi, np.pi / 3)
    phase=np.arange(0,np.pi/2,np.pi/10)
    compute_tensor=0
    
    bg=1
    if compute_tensor==1:
        for nb in range(1, raster.RasterCount+1):
            begin=1
            print nb
            
            NIRb=raster.GetRasterBand(nb)
            NIR=NIRb.ReadAsArray()#.astype('double')
            
            rows_a,cols_a=NIR.shape
            rows_a=int(rows_a/block_size)*block_size
            cols_a=int(cols_a/block_size)*block_size
            
            mask = np.greater(np.max(NIR), 0)
            ratio=np.choose(mask,(0,(NIR)))
            # ratio=ratio#+(nb-1)*200
            
            
            ratio=ratio[0:rows_a,0:cols_a]
#            print np.min(ratio)
#            print np.max(ratio)
            #ratiot=np.choose(mask,(0,(NIR/np.max(NIR))))
            # ratiot=ratiot[0:rows_a,0:cols_a]
            # print bands_all.shape
            bands_all.append(ratio)
            # bands_all.append(np.power(ratio,2))
           
#            for b in range(nb+1,raster.RasterCount+1):
#                bb=raster.GetRasterBand(b)
#                bb=bb.ReadAsArray().astype('double')
#                mask = np.greater(np.max(bb), 0)
#                bbtemp=np.choose(mask,(0,(bb/np.max(bb))))
#                bbtemp=bbtemp[0:rows_a,0:cols_a]
#                # bands_all.append(ratiot*bbtemp)
#                mask = np.greater(np.fabs(NIR+bb), 0)
##                bb=bb[0:rows_a,0:cols_a]
##                NIR=NIR[0:rows_a,0:cols_a]
#                ratio=np.choose(mask,(0,np.arctan((-NIR+bb)/(NIR+bb))))
#                ratio=(ratio-np.min(ratio))/(np.max(ratio)-np.min(ratio))
#                # ratio=NIR+bb
##                if bg==1:
##                    tt=ratio
##                    bg=0
##                else:
##                    tt=tt+bb
##                ratio=ratio.astype(np.float32)
#                
#                
##                gab1=get_gabor(ratio,ksize,ksize/4,ksize/2,moves,slopes,rows_a,cols_a)###single_line
##                gnum,rr,ss=gab1.shape
##                bg=1
##                timg=ratio[0:rr,0:ss]
##                for g in range(gnum):
##                
##                    timg+=gab1[g,:,:]
##                timg=timg[0:rows_a,0:cols_a]
## bands_all.append(timg)
#            
#                ratio=ratio[0:rows_a,0:cols_a]
#                #  tt=tt[0:rows_a,0:cols_a]
#                # ratio=ratio[1:3000,1:3000]
##                
##                #s   bands_all.append(ratio+2)
## bands_all.append(ratio)
        n_comp=6
        
        bands_all=np.asarray(bands_all)###we want size of pixelsize*(self+neighbors)*spectral
        T = bands_all
        print T.shape
        sio.savemat('tensor_DC.mat',{'T':T})
        # sio.savemat('tensor_bands1.mat',{'T':T})
        hhhh
    begin=1       
    for b in range(1,raster.RasterCount+1):
        bb=raster.GetRasterBand(b)
        bb=bb.ReadAsArray().astype('double')
        rows_a,cols_a=bb.shape
        rows_a=int(rows_a/block_size)*block_size
        cols_a=int(cols_a/block_size)*block_size
        
        bb=bb[0:rows_a,0:cols_a]
        # bb=bb[1:3000,1:3000]
        if begin==1:
            temp=bb
            begin=0
        else:
            temp=temp+bb
    inds=(temp==0)
    mat_data=(sio.loadmat('sparse_pavia.mat'))
    #  mat_data=(sio.loadmat('pavia.mat'))
# mat_data=(sio.loadmat('aviris3.mat'))
# mat_data=(sio.loadmat('sparse_india_www.mat'))
#  mat_data=(sio.loadmat('nias_shift_im1.mat'))
    # mat_data=(sio.loadmat('india_block_www.mat'))
    #mat_data=(sio.loadmat('sparse_1.mat'))
    # mat_data=(sio.loadmat('nias_shift_rat_class1_block16.mat'))
                           
    bands_all=mat_data['total_comp']
    
#    mat_data=(sio.loadmat('components_16sh_6c.mat'))
#    bands_all2=mat_data['total_comp']
#    print bands_all.shape
#    print bands_all2.shape
    
   
#    comp_num=mat_data['comps'] ##number of factor components in each class
#    class_num=mat_data['bcount']##number of block sies in tensor program(partitions into classes
    comp_num=3#comp_num[0][0]-1#temporarily -1
    class_num=0#class_num[0][0]
#    print class_num
#    hhh
    bands_all=bands_all[:,:,0:9]
    rows_a, cols_a, n_bands_all = bands_all.shape
  
    rangef=np.asarray(np.asarray(range(block_size)))
    print rangef.shape
    print bands_all.shape
    
#    for b in range(n_bands_all):
#        bands_all[inds,b]=100
    comp_num=3
#    for i in range(n_bands_all):
##        if i in [3,4,8,9,13,14,18,19]:
##            continue
#        
#        bands_all[inds,i]=100
    # bands_all2[inds,i]=100
          
        
#    for i in range(comp_num*class_num):
#        if i%(comp_num+1)>3:
#            continue

    
    
#    bands_all=np.delete(bands_all,rangef,axis=0)
#    xrange=range(bands_all.shape[0]-block_size,bands_all.shape[0])
#    bands_all=np.delete(bands_all,xrange,axis=0)
#    
#    bands_all=np.delete(bands_all,rangef,axis=1)
#    yrange=range(bands_all.shape[1]-block_size,bands_all.shape[1])
#    bands_all=np.delete(bands_all,yrange,axis=1)
#    print bands_all.shape
     
#  bands_all=bands_all[:,:,]
#    
#    inds=[4,5,6]
#    bands_all=bands_all[:,:,inds]
    rows_a, cols_a,n_bands_all = bands_all.shape

#    n_bands_all=n_bands_all[:,:,inds]
# bands_all=bands_all[:,:,0:2]
    bands_all=bands_all[block_size:rows_a-block_size,block_size:cols_a-block_size,:]
    rows_a, cols_a,n_bands_all = bands_all.shape
    print bands_all.shape
    flat_pixels_all = bands_all.reshape((rows_a*cols_a,n_bands_all))

#    for i in range(comp_num):
#        #        if i in [3,4,8,9,13,14,18,19]:
#        #            continue
#        # if i==arg_num:
#        btemp=bands_all[:,:,i+comp_num*i]
#        flat_pixels_all = btemp.reshape((rows_a*cols_a,n_bands_all))
#            break
    
    

#  flat_pixels_all=flat_pixels_all[:,0:2]
    flat_pixels_all=np.transpose(flat_pixels_all)

#    rows_a, cols_a, n_bands_all = bands_all2.shape
#
#    bands_all2=np.delete(bands_all,rangef,axis=0)
#    xrange=range(bands_all.shape[0]-block_size,bands_all2.shape[0])
#    bands_all2=np.delete(bands_all,xrange,axis=0)
#    
#    bands_all2=np.delete(bands_all,rangef,axis=1)
#    yrange=range(bands_all.shape[1]-block_size,bands_all2.shape[1])
#    bands_all2=np.delete(bands_all,yrange,axis=1)
#    rows_a, cols_a, n_bands_all = bands_all2.shape
#
#    for i in range(n_bands_all):
#        #  bands_all2[inds,i]=100
##        print flat_pixels_all.shape
##        print "new"
##        print bands_all2[:,:,i].shape
#        print flat_pixels_all.shape
#        flat_pixels_all=np.append(flat_pixels_all,bands_all2[:,:,i].reshape(1,rows_a*cols_a),axis=0)
#
#    mat_data=(sio.loadmat('components_64_11c.mat'))
#    bands_all2=mat_data['total_comp']
#    rows_a, cols_a, n_bands_all = bands_all2.shape
#    for i in range(comp_num*class_num):
#        if i%(comp_num+1)>3:
#            continue
#        bands_all2[inds,i]=100
#        print flat_pixels_all.shape
#        print "new"
#        print bands_all2[:,:,i].shape
#flat_pixels_all=np.append(flat_pixels_all,bands_all2[:,:,i].reshape(1,rows_a*cols_a),axis=0)
    bands_all=[]
    # flat_pixels_all=[]
#    for nb in range(1, raster.RasterCount+1):
#        begin=1
#            
#        bg=1
#        NIRb=raster.GetRasterBand(nb)
#        NIR=NIRb.ReadAsArray().astype('double')
#
#        if nb==3:
#            break
##        NIR=np.delete(NIR,rangef,axis=0)
##        NIR=np.delete(NIR,xrange,axis=0)
##
##        NIR=np.delete(NIR,rangef,axis=1)
##        NIR=np.delete(NIR,yrange,axis=1)
#
#        mask = np.greater(np.max(NIR), 0)
#      
#        for b in range(nb+1,raster.RasterCount+1):
#            rows_a,cols_a=NIR.shape
#            print NIR.shape
#            
#            print block_size
#            rows_a=int(rows_a/block_size)*block_size
#            cols_a=int(cols_a/block_size)*block_size
#            bb=raster.GetRasterBand(b)
#            bb=bb.ReadAsArray().astype('double')
##            bb=np.delete(bb,rangef,axis=0)
##            bb=np.delete(bb,xrange,axis=0)
##        
##            bb=np.delete(bb,rangef,axis=1)
##            bb=np.delete(bb,yrange,axis=1)
#            mask = np.greater(np.fabs(NIR+bb), 0)
#            ratio=np.choose(mask,(0,np.arctan((-NIR+bb)/(NIR+bb))))
#            ratio=(ratio-np.min(ratio))/(np.max(ratio)-np.min(ratio))
#            
#            ratio=ratio[0:rows_a,0:cols_a]
#            
#            rows_a,cols_a=ratio.shape
#            ratio=ratio[block_size:rows_a-block_size,block_size:cols_a-block_size]
#            print "rrrrr"
#            print ratio.shape
#            rows_a,cols_a=ratio.shape
#        
#        #  ratio[inds]=100
#            
#            # ratio[inds]=100##assign constant to cut values
#            
#            #  print (flat_pixels_all.shape)
##bands_spectral.append(ratio)
#            bands_all.append(ratio)
#            
#            flat_pixels_all=np.append(flat_pixels_all,ratio.reshape(1,rows_a*cols_a),axis=0)
#            if b==3:
#                break

#        


    flat_pixels_all=np.transpose(np.asarray(flat_pixels_all))
    
    ###########
#    bands_all=np.dstack(bands_all)
#    rows, cols, n_bands = bands_all.shape
#    # A sample is a vector with all the bands data. Each pixel (independent of its position) is a
#    # sample.
#    n_samples = rows*cols
#    flat_pixels_all = bands_all.reshape((n_samples, n_bands))
    ############
    
    print flat_pixels_all.shape
#   s_inds=[0,1,3,4,6,7,9,10,11,12]
#   flat_pixels_all=flat_pixels_all[:,s_inds]
    #  flat_pixels_all=flat_pixels_all.reshape(-1, 1)
    #  bands_all=np.append(bands_all,bands_all2[:,:,i])
#    comp_num=comp_num*class_num
    class_num=0
    print comp_num
    print class_num
#    print comp_num.dtype
# hhhh

#        #  print bands_all.shape
#        
#        print np.min(bands_all[:,:,i])
#        print np.max(bands_all[:,:,i])
#        cv2.imwrite('tensor_'+str(i)+'.png',(bands_all[:,:,i]-np.min(bands_all[:,:,i]))*100)
#    hhh
    

#samples, tensor_comps=flat_pixels_all.shape
#    print samples
#    print tensor_comps
   
   
    #  inds=[1,2,7,8,13,14,19,20,25,26,31,32,37,38]
    # flat_pixels_all=flat_pixels_all[1:8,:]
    
#  flat_pixels_texture=flat_pixels_all[1:comp_num,:]
#   flat_pixels_all=np.transpose(flat_pixels_all)

#    for nb in range(1, raster.RasterCount+1):
#        begin=1
#            
#        bg=1
#        NIRb=raster.GetRasterBand(nb)
#        NIR=NIRb.ReadAsArray().astype('double')
#        rows_a,cols_a=NIR.shape
#        rows_a=int(rows_a/block_size)*block_size
#        cols_a=int(cols_a/block_size)*block_size
#        #        mask = np.greater(np.max(NIR), 0)
#        #        ratio=np.choose(mask,(0,(NIR/np.max(NIR))))
#        #        ratio=ratio[0:rows_a,0:cols_a]
#        # bands_all.append(ratio)
#        
#        for b in range(nb+1,raster.RasterCount+1):
#            bb=raster.GetRasterBand(b)
#            bb=bb.ReadAsArray().astype('double')
#            mask = np.greater(np.fabs(NIR+bb), 0)
#            ratio=np.choose(mask,(0,np.arctan((-NIR+bb)/(NIR+bb))))
#            ratio=ratio[1:rows_a+1,1:cols_a+1]
#            ratio[inds]=100##assign constant to cut values
#            
#            print (flat_pixels_all.shape)
#            # bands_spectral.append(ratio)
#            flat_pixels_all=np.append(flat_pixels_all,ratio.reshape(1,rows_a*cols_a),axis=0)

    
#    flat_pixels_all=np.transpose(np.asarray(flat_pixels_all))
    return class_num,comp_num,[],[],flat_pixels_all,rows_a,cols_a,block_size

    snum,bnum=flat_pixels_all.shape

#print tensor_comps
    print bnum

    probs=GMM(n_components=sample_cl, covariance_type='diag').fit(flat_pixels_all).predict_proba(flat_pixels_all)#[:,tensor_comps+1:bnum])###partition based on spectras
    result=np.argmax(probs,1)
    samples_s=[]
    samples_t=[]
    samples_t2=[]
    samples_t3=[]
    samples_t4=[]
    samples_real=[]
    total_inds=[]
    rand_inds=[]
    begin=1
    for i in range(sample_cl):
        inds=(result==i )
        
        cluster_sample_num=4000
#        
#        points_spec=flat_pixels_spectral[inds]
#        points_text=flat_pixels_texture[inds]
#        points_text2=flat_pixels_texture2[inds]
#        points_text3=flat_pixels_texture3[inds]
#        points_text4=flat_pixels_texture4[inds]
        points=flat_pixels_all[inds]
        if points.shape[0]<=5:
            continue
        vars_label=np.sqrt(np.var(points[:,0]))
        print "variance"
        print vars_label
        if vars_label==0:
            cluster_sample_num=10
#        else:
#            print "variance"
#            print vars_label
#            cluster_sample_num=int(math.floor(cluster_sample_num/(1-vars_label)))
      
        
        
        if points.shape[0]<cluster_sample_num:
            cluster_sample_num=points.shape[0]-2
        rand_inds=(random.sample(range(1, points.shape[0]-1),cluster_sample_num))
        print (np.asarray(rand_inds)).shape
        if begin==1:
            total_inds=(np.asarray(rand_inds))
            begin=0
        else:
            total_inds=np.append(total_inds,np.asarray(rand_inds),axis=0)
        print "shape"
        print total_inds.shape
# total_inds.append(np.transpose(np.asarray(rand_inds)))
        for r in rand_inds:
#            samples_s.append(points_spec[r,:])
#            samples_t.append(points_text[r,:])
#            samples_t2.append(points_text2[r,:])
#            samples_t3.append(points_text3[r,:])
#            samples_t4.append(points_text4[r,:])
            samples_real.append(points[r,:])
    
        

#####now partition based on texture
#    probs=GMM(n_components=sample_cl, covariance_type='diag').fit(flat_pixels_texture).predict_proba(flat_pixels_texture)###partition based on spectras
#    result=np.argmax(probs,1)
#    total_inds=[]
#    rand_inds=[]
#    for i in range(sample_cl):
#        inds=(result==i )
#        
#        cluster_sample_num=1000
#        
#        points_spec=flat_pixels_spectral[inds]
#        points_text=flat_pixels_texture[inds]
#        points_text2=flat_pixels_texture2[inds]
#        points_text3=flat_pixels_texture3[inds]
#        points_text4=flat_pixels_texture4[inds]
#        points=flat_pixels_all[inds]
#
#        if points_text.shape[0]<=5:
#            continue
#        if points_text.shape[0]<cluster_sample_num:
#            cluster_sample_num=points_text.shape[0]-2
#        rand_inds=(random.sample(range(1, points_text.shape[0]-1),cluster_sample_num))
#        total_inds.append(rand_inds)
#        for r in rand_inds:
#            samples_s.append(points_spec[r,:])
#            samples_t.append(points_text[r,:])
#            samples_t2.append(points_text2[r,:])
#            samples_t3.append(points_text3[r,:])
#            samples_t4.append(points_text4[r,:])
#            samples_real.append(points[r,:])

         
#                
#    samples_s=np.asarray(samples_s)
#    samples_t=np.asarray(samples_t)
#    samples_t2=np.asarray(samples_t2)
#    samples_t3=np.asarray(samples_t3)
#    samples_t4=np.asarray(samples_t4)
#samples_real=np.asarray(samples_real)
#    total_inds=(np.dstack(total_inds))
#    t1,t2,t3=total_inds.shape
#    total_inds=np.transpose(total_inds.reshape(1,t2*t3))
#    total_inds=total_inds[:,0]
#
#    print total_inds.shape
# total_inds=np.dstack(total_inds)
#total_inds=np.asarray(total_inds)
    return class_num,comp_num,total_inds,samples_real,flat_pixels_all,rows_a,cols_a,block_size
#    plt.hist(fIndex, bins='auto') 
#    plt.title("Histogram with 'auto' bins")
#    plt.show()
#    hhhh
    


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """
    Rasterize, in a single image, all the vectors in the given directory.
    The data of each file will be assigned the same pixel value. This value is defined by the order
    of the file in file_paths, starting with 1: so the points/poligons/etc in the same file will be
    marked as 1, those in the second file will be 2, and so on.
    :param file_paths: Path to a directory with shapefiles
    :param rows: Number of rows of the result
    :param cols: Number of columns of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i+1
        logger.debug("Processing file %s: label (pixel value) %i", path, label)
        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection,
                                     target_value=label)
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray()
        logger.debug("Labeled pixels: %i", len(a.nonzero()[0]))
        labeled_pixels += a
        ds = None
    return labeled_pixels


def write_geotiff(fname, data, geo_transform, projection, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)

    ct = gdal.ColorTable()
    for pixel_value in range(classes):
        print pixel_value
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        # 'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return


def report_and_exit(txt, *args, **kwargs):
    logger.error(txt, *args, **kwargs)
    exit(1)


if __name__ == "__main__":
    
    neighbors=5
    
    opts = docopt(__doc__)
    ###class_number: giving the number of the class that needs cleaning up, -1 for running new
    raster_data_path = opts["<input_fname>"]
   
    train_data_path = opts["<train_data_path>"]
    output_fname = opts["<output_fname>"]
    class_number=int(opts["<class_number>"])-1
    class_number_correct=int(opts["<class_number_correct>"])-1
    validation_data_path = opts['--validation'] if opts['--validation'] else None
    log_level = logging.DEBUG if opts["--verbose"] else logging.INFO
    method = opts["--method"]

    logging.basicConfig(level=log_level, format='%(asctime)-15s\t %(message)s')
    gdal.UseExceptions()
    img=gdal.Open(raster_data_path)
    inputArray=img.ReadAsArray()
    
    try:
        raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
        # landsat_dataset = gdal.Open(landsat_path, gdal.GA_ReadOnly)
        #cloud_dataset = gdal.Open(cloud_path, gdal.GA_ReadOnly)
    
    except RuntimeError as e:
        report_and_exit(str(e))
    
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
 
    logger.debug("Reading the input: %s", raster_data_path)

    classes=9
    if class_number>100:


        bands_data = []

        logger.debug("Process the training data")
            #  for i in range(15):

#class_num,comp_num,total_inds,samples_real,flat_pixels_all,rows_a,cols_a
        class_num,comp_num,rand_inds,samples,flat_pixels_all,rows,cols,block_size= get_samples(raster_dataset,inputArray,1,1)
#        print "GMM ifrst"
#        probs=GMM(n_components=classes, covariance_type='diag').fit(flat_pixels_all).predict_proba(flat_pixels_all)#[:,tensor_comps+1:bnum])###partition based on spectras
#        result=np.argmax(probs,1)
#        members=result+1
#        classification = members.reshape((rows, cols))
#        # sio.savemat('First_run_data.mat',{'labels':labels,'inds_total':inds_total,'samples':samples_spec})
#        write_geotiff(output_fname, classification, geo_transform, proj)
#        hhh
        print "KMeans start"
        result=KMeans(n_clusters=classes,n_init=50).fit(flat_pixels_all).labels_
        block_size=int(block_size)
        # block_size=8
    
        members=result+1
    
        classification = members.reshape((rows, cols))
        classification=np.pad(classification, (block_size,block_size), 'edge')
#        # sio.savemat('First_run_data.mat',{'labels':labels,'inds_total':inds_total,'samples':samples_spec})
        write_geotiff(output_fname, classification, geo_transform, proj)
        gggg
#        hhh
        samples=np.asarray(samples)
        total_bands=(flat_pixels_all.shape)[1]
        print "nu start"
        print total_bands
        print comp_num*class_num
            # print class_num
        fit_samples=samples[:,comp_num*class_num:total_bands]
        samples_spectral=SpectralClustering(n_clusters=classes,affinity='nearest_neighbors',n_neighbors=10,assign_labels='discretize').fit(fit_samples)
        amatrix=(samples_spectral.affinity_matrix_).toarray()

#        for i in range(class_num):####goes over different block sizes
#            print i*comp_num
#            print i*comp_num+comp_num
##            inds=arange(i,class_num,)###start, stop, step. We want indices over same band with same component el
#            ###-> step size should be class size
#            last_el=min((i+1)*2,comp_num)###get fewer elements for smaller block sizes
#            samples_temp=flat_pixels_all[:,i*comp_num:i*comp_num+comp_num]
#            
#            samples_temp=samples_temp[rand_inds,:]
#            print samples_temp.shape
#            samples_part=SpectralClustering(n_clusters=classes,affinity='nearest_neighbors',n_neighbors=6,assign_labels='discretize').fit(samples_temp)
#            amatrix=(amatrix*(samples_part.affinity_matrix_).toarray())

        
        
        
    #            # sio.savemat('matrices.mat',{'Amat':amatrix,'count_n':count_n,'sum_n':sum_n})
#        while (np.count_nonzero(amatrix)>amatrix.shape[0]*(neighbors**2)):
#            print np.count_nonzero(amatrix)
#            means=amatrix.mean(1)
#            maxs=np.amax(amatrix,axis=1)
#            maxs_extend=maxs*np.ones(shape=(1, amatrix.shape[0]))
#            means_extend=means*np.ones(shape=(1, amatrix.shape[0]))
#            substract=means_extend+(maxs_extend-means_extend)*0.5
#            amatrix=amatrix-substract
#            neg_val=amatrix<0
#            amatrix[neg_val]=0
        
        labels=samples_spectral.labels_
#        samples_total=SpectralClustering(n_clusters=classes,affinity='precomputed',n_neighbors=flat_pixels_all.shape[1],assign_labels='discretize')
#        samples_total.fit_predict(amatrix)
#        labels=samples_total.labels_
        print "before NN"
        neigh = KNeighborsClassifier(n_neighbors=10)
        neigh.fit(samples, labels)

        result=neigh.predict(flat_pixels_all)
        # result=np.lib.pad(result, ((block_size, block_size), (block_size, block_size)), 'edge')
#result=KMeans(n_clusters=classes).fit(flat_pixels).labels_
        members=result+1
        classification = members.reshape((rows, cols))
        classification=np.pad(classification, (block_size,block_size), 'edge')
        # sio.savemat('First_run_data.mat',{'labels':labels,'inds_total':inds_total,'samples':samples_spec})
        write_geotiff(output_fname, classification, geo_transform, proj)
  
