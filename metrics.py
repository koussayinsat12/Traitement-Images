import numpy as np
import cv2 as cv
class Metrics():
    
    def __init__(self,img,new_img):
        self.img=img
        self.new_img=new_img
        
    def MSE(self):
        return np.mean(np.square(self.img-self.new_img))

    def PSNR(self):
        return (10*np.log10(255)**2)/self.MSE()
    
    def SD(self):
        height=self.new_img.shape[0]
        width=self.new_img.shape[1]
        X=np.mean(self.new_img)
        cum_sum=0
        for i in range(height):
            for j in range(width):
                cum_sum=cum_sum+np.square(self.new_img[i,j]-X)
        return np.sqrt((1/((width*height)-1))*cum_sum)


                
                
                 