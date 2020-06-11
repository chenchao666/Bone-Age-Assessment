import cv2
import numpy as np
import os


    

def crop(img,mask):
    index = np.where(mask>0)
    top =np.min(index[0])
    bottom = np.max(index[0]) 
    left = np.min(index[1])
    right = np.max((index[1]))
   # extract hand region
   # if top > 200:
   #     top =top -200
   # elif top > 100:
   #     top = top -100

   # extract region1
   # if left>100:
   #     left=left-70

    croped_img = img[top:bottom,left:right]
    return croped_img



def maskout(img,mask):
    index = np.where(mask>0)
    top =np.min(index[0])
    bottom = np.max(index[0]) 
    left = np.min(index[1])
    right = np.max((index[1]))
    img[top:bottom,left:right]=np.random.randint(255)
    return img


def find_max_component(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_ind = np.argmax(area)
    print (area)
    for ind in range(len(contours)):
        if ind != max_ind:
            cv2.fillConvexPoly(mask,contours[ind],0)
    return mask




if __name__=="__main__":
    path_list = os.listdir('train/')
    path_list.sort()
    print (path_list)
    kernel = np.ones((5,5),np.uint8)
    for path in path_list:
        img = cv2.imread('train/'+path,0)
        heatmap = cv2.imread('GAPHandAttention/heatmap/'+path,0)
        ret,mask =  cv2.threshold(heatmap,40,255,cv2.THRESH_BINARY)
        mask = find_max_component(mask)
        # mask = cv2.dilate(mask,kernel,iterations=1)
        # cv2.imwrite('patches/'+path,mask)
        # img = img*mask
        # cv2.imwrite('patches/'+path,img)


        print (path)
        croped_img= crop(img,mask)
        cv2.imwrite('Hand/'+path,croped_img)

        #MaskImg = maskout(img,mask)
        #cv2.imwrite('Maskout/'+path,MaskImg)


