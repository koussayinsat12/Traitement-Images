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

def apply_clipping(histogram, clip_limit):
    clipped_histogram = np.clip(histogram, 0, clip_limit)
    excess = int(np.sum(histogram - clipped_histogram))
    redistribute = excess // 256
    clipped_histogram += redistribute
    excess -= redistribute * 256
    for i in range(excess):
        clipped_histogram[i % 256] += 1
    return clipped_histogram

def calculate_dynamic_ranges(cdf, m1, m2, m3):
    L = 256
    total_pixels = cdf[-1]
    spans = [m1, m2 - m1, m3 - m2, total_pixels - m3]
    range_starts = np.zeros(4, dtype=int)
    range_ends = np.zeros(4, dtype=int)
    sum_spans = sum(spans)
    for i in range(4):
        range_i = (L - 1) * spans[i] // sum_spans
        range_starts[i] = range_ends[i-1] + 1 if i > 0 else 0
        range_ends[i] = range_starts[i] + range_i
    return range_starts, range_ends


def apply_QDHE(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    m1, m2, m3 = (np.percentile(cdf, q) for q in [25, 50, 75])

    clip_limit = np.mean(image)
    clipped_hist = apply_clipping(hist, clip_limit)
    clipped_cdf = clipped_hist.cumsum()

    range_starts, range_ends = calculate_dynamic_ranges(clipped_cdf, m1, m2, m3)

    equalized_image = np.zeros_like(image)
    for i in range(4):
        mask = (image >= range_starts[i]) & (image <= range_ends[i])
        sub_image = image[mask]
        
        sub_hist, _ = np.histogram(sub_image, bins=np.arange(256), range=(0, 255))
        sub_cdf = sub_hist.cumsum()
        sub_cdf_normalized = sub_cdf * (range_ends[i] - range_starts[i]) / sub_cdf[-1] + range_starts[i]
        equalized_image[mask] = sub_cdf_normalized[sub_image - range_starts[i]]
    
    return equalized_image

    
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









#plt.imshow(QDHE(hist,img,6000,bin_centers=bins),cmap="Greys_r")
#plt.imshow(CHE(hist,img,bin_centers=bins),cmap="Greys_r")
#plt.show()
#seperations,_,_=segment_histogram(hist,img.shape[0],img.shape[1])
#plot_histogram_with_separation(hist, seperations)
