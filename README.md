# Image_stitching_from_scratch

This program performs image stitching for more than two images and is written from scratch including calculation of Homography and RANSAC algorithm.                               

### Requirements:
pip install opencv-contrib-python

### Folder structure:
* src - contains the source code                                                             
* data - contains the images to be stitched


To run the code from src direcctory, open a terminal and enter the following:
```sh
python stitch.py <folder_path>      
```
example:                                                                                     
```sh
python stitch.py ../data/mountain 
```


### Note:
* You need to remove the panaroma.jpg file from folders to run the code  
* install opencv-contrib-python as cv::xfeatures2d::SIFT_create has been deprecated in higher versions of opencv

### Sample Output:

#### Input Images:

![alt text](https://github.com/axay15/Image_stitching_from_scratch/blob/master/data/mountain/mountain1.jpg)    ![alt text](https://github.com/axay15/Image_stitching_from_scratch/blob/master/data/mountain/mountain2.jpg) 


#### Output Panorama:

![alt text](https://github.com/axay15/Image_stitching_from_scratch/blob/master/data/mountain/panorama.jpg) 
