import numpy as np
import cv2 as cv  
from matplotlib import pyplot as plt
import numpy as np 
from skimage import data 
from skimage.util import img_as_ubyte 
from skimage import exposure 
import skimage.morphology as morp 
from skimage.filters import rank 
from skimage.io import imsave, imread

#plot
def plot_histogram_with_separation(image_histogram, separating_points):
    plt.figure(figsize=(8, 4))  # Set the figure size
    plt.bar(range(len(image_histogram)), image_histogram, color='skyblue') 
    plt.title('Image Histogram with Separating Points') 
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)  
    for separating_point in separating_points:
        plt.axvline(x=separating_point, color='red', linestyle='--', linewidth=1)

    plt.show()  

def clip_histogram(image_histogram, Tc):
    clipped_histogram = [min(bin_count, Tc) for bin_count in image_histogram]
    return clipped_histogram

def segment_histogram(image_histogram, width, height):
    total_pixels = width * height
    m1 = 0.25
    m2 = 0.5
    m3 = 0.75
    pixels_m1 = int(m1 * total_pixels)
    pixels_m2 = int(m2 * total_pixels)
    pixels_m3 = int(m3 * total_pixels)
    separating_points = [0, 0, 0, 0, 255]
    cumulative_sum = 0
    for i, bin_count in enumerate(image_histogram):
        cumulative_sum += bin_count
        if cumulative_sum >= pixels_m1 and separating_points[1] == 0:
            separating_points[1] = i
        if cumulative_sum >= pixels_m2 and separating_points[2] == 0:
            separating_points[2] = i
        if cumulative_sum >= pixels_m3 and separating_points[3] == 0:
            separating_points[3] = i
            break
    sub_histograms = []
    start_index = 0
    for point in separating_points:
        if (point != 0 and point != 255):
            end_index = point
            sub_histogram = image_histogram[start_index:end_index]
            sub_histograms.append(sub_histogram)
            start_index = end_index
    sub_histogram = image_histogram[start_index:]
    sub_histograms.append(sub_histogram)
    pixels_m = [0, pixels_m1, pixels_m2, pixels_m3, total_pixels]
    return separating_points, sub_histograms, pixels_m

def QDHE(image_histogram,img,Tc,bin_centers):
    height=img.shape[0]
    width=img.shape[1]
    separating_points, sub_histograms, pixels_m = segment_histogram(image_histogram=image_histogram, width=width, height=height)
    total_pixels = width * height
    total_span = np.sum(pixels_m)
    new_hist = []
    for i, sub_histogram in enumerate(sub_histograms):
        sub_histogram=clip_histogram(sub_histogram,Tc)
        min_val = separating_points[i]
        max_val = separating_points[i+1]
        span = abs(int(i/4 * total_pixels) - int((i+1)/4 * total_pixels))
        
        range_val = int(255 * span / total_span)
        
        new_min = max_val + 1
        new_max = new_min + range_val
        
        sub_histogram = np.round((np.cumsum(sub_histogram)/total_pixels) * (new_max - new_min)) + new_min
        
        new_hist.extend(sub_histogram.astype(int))
    cdf=np.array(new_hist)
    print(cdf.shape)
    out = np.interp(img.flatten(), bin_centers, cdf).astype(int)
    out = out.reshape(img.shape)

        
    return out

    
def CHE(hist_img,img,bin_centers):
    total_pixels=img.shape[0]*img.shape[1]
    cum_sum=hist_img.cumsum()
    cdf=np.round(
        ((cum_sum-np.min(cum_sum))/(total_pixels-np.min(cum_sum)))*255
    ).astype(int)
    print(cdf)
    out = np.interp(img.flatten(), bin_centers, cdf).astype(int)
    out = out.reshape(img.shape)
    return out 









path = "vallee.png"
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hist,bins = np.histogram(img.flatten(),256,[0,256])
bins=np.array(bins[:256])

#plt.imshow(QDHE(hist,img,6000,bin_centers=bins),cmap="Greys_r")
#plt.imshow(CHE(hist,img,bin_centers=bins),cmap="Greys_r")
#plt.show()
#seperations,_,_=segment_histogram(hist,img.shape[0],img.shape[1])
#plot_histogram_with_separation(hist, seperations)
