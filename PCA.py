#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
import cv2


# In[7]:


#Calculating PCA
def pca(immatrix):
    #Mean Calculation
    MatMean = immatrix.mean(axis=0) 
    
    #Center all the images using mean
    matrix = immatrix - MatMean 

    #PCA Calculation 
    #Calculating Covariance matrix
    M = np.dot(matrix,matrix.T) 
    
    #Finding eigenvalues and eigenvectors
    EVal,EVec = np.linalg.eigh(M) 
    
    #calculate PCA components
    tmp = np.dot(matrix.T,EVec).T
    V = tmp[::-1]
    return V,matrix,EVal,EVec,MatMean


# In[8]:


#Reconstructing images using pca and mean
#Ref: https://github.com/NourozR/Reconstruction-and-Compression-of-Color-Images/blob/master/image_reconstruction_using_PCA.py
def recon(k,EVec,EVal,immatrix,mean_X):
    #Sorting according to eigenvalues
    p = np.size(EVec, axis =1)
    idx = np.argsort(EVal) 
    idx = idx[::-1] 
    eig_vec = EVec[:,idx] 
    eig_val = EVal[idx] 
    
    #Reconstructing Image
    eig_vec = eig_vec[:, range(k)]
    score = np.dot(eig_vec.T, immatrix)
    mean = mean_X.T
    prod = np.dot(eig_vec, score)
    recon =  mean + prod 
    return recon


# In[9]:


f, axarr = plt.subplots(4,2,constrained_layout=True,figsize=(15, 12))
elem = './Images/elephant.png'
print("Reading image "+elem)

img = cv2.imread(elem)

img = color.rgb2gray(img)#converting image into grayscale

axarr[2,0].set_title('Original unblocked Image')
axarr[2,0].imshow(img.reshape(640,640),cmap='gray')
io.imsave("./Images/S.png",img)

#dividing image into 16x16 non-overlapping patches
blocks = np.array([img[i:i+16, j:j+16] for i in range(0, img.shape[1], 16) for j in range(0, img.shape[0], 16)])

m=16
n=16
          
            
#storing all flattened images            
immatrix = np.array([(im).flatten() for im in blocks],'f')

#calculating PCA
V,immatrix,e,EV,mean_X = pca(immatrix)

#storing 16x16 images the first ten principal components
f,err = plt.subplots(5,2,constrained_layout=True,figsize=(15, 12))

err[0,0].set_title('PCA1 Image')
err[0,0].imshow(V[0].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA1.png",V[0].reshape(m,n))

err[0,1].set_title('PCA2 Image')
err[0,1].imshow(V[1].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA2.png",V[0].reshape(m,n))

err[1,0].set_title('PCA3 Image')
err[1,0].imshow(V[2].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA3.png",V[0].reshape(m,n))

err[1,1].set_title('PCA4 Image')
err[1,1].imshow(V[3].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA4.png",V[0].reshape(m,n))

err[2,0].set_title('PCA5 Image')
err[2,0].imshow(V[4].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA5.png",V[0].reshape(m,n))

err[2,1].set_title('PCA6 Image')
err[2,1].imshow(V[5].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA6.png",V[0].reshape(m,n))

err[3,0].set_title('PCA7 Image')
err[3,0].imshow(V[6].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA7.png",V[0].reshape(m,n))

err[3,1].set_title('PCA8 Image')
err[3,1].imshow(V[7].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA8.png",V[0].reshape(m,n))

err[4,0].set_title('PCA9 Image')
err[4,0].imshow(V[8].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA9.png",V[0].reshape(m,n))

err[4,1].set_title('PCA10 Image')
err[4,1].imshow(V[9].reshape(m,n),cmap='gray')
io.imsave("./Images/PCA10.png",V[0].reshape(m,n))

#reconstructing image patches
k = 10 #Number of Principal Components
recon_img_mat = recon(k,EV,e,immatrix,mean_X) #Call reconstruction function
k = 100 #Number of Principal Components
recon_img_mat1 = recon(k,EV,e,immatrix,mean_X) #Call reconstruction function
k = 1600 #Number of Principal Components
recon_img_mat2 = recon(k,EV,e,immatrix,mean_X) #Call reconstruction function


#original Image
axarr[0,0].set_title('Original Image')
axarr[0,0].imshow(immatrix[0].reshape(m,n),cmap='gray')
io.imsave("./Images/S_16x16.png",immatrix[0].reshape(m,n))

#Shown reconstructed image patch using 10 principal components
axarr[0,1].set_title('Reconstructed(p=10) Image')
axarr[0,1].imshow(recon_img_mat[0].reshape(m,n),cmap='gray')
io.imsave("./Images/R10_16x16.png",recon_img_mat[0].reshape(m,n))

#Shown reconstructed image patch using 100 principal components
axarr[1,0].set_title('Reconstructed(p=100) Image')
axarr[1,0].imshow(recon_img_mat1[0].reshape(m,n),cmap='gray')
io.imsave("./Images/R100_16x16.png",recon_img_mat1[0].reshape(m,n))

#Shown reconstructed image patch using 1600 principal components
axarr[1,1].set_title('Reconstructed(p=1600) Image')
axarr[1,1].imshow(recon_img_mat2[0].reshape(m,n),cmap='gray')
io.imsave("./Images/R1600_16x16.png",recon_img_mat2[0].reshape(m,n))

#reconstructing image for k=10
dempo1 = recon_img_mat.reshape(1600,16,16)

fin = 0;
img2 = np.zeros((640,640),dtype=float)
for i in range(0,640,16):
    for j in range(0,640,16):
        img2[i:i+16, j:j+16] = dempo1[fin]
        fin=fin+1;
    

axarr[2,1].set_title('Reconstructed unblocked Image(p=10)')
axarr[2,1].imshow(img2,cmap='gray')
io.imsave("./Images/R10.png",img2)

#reconstructing image for k=100
dempo = recon_img_mat1.reshape(1600,16,16)

fin = 0;
img1 = np.zeros((640,640),dtype=float)
for i in range(0,640,16):
    for j in range(0,640,16):
        img1[i:i+16, j:j+16] = dempo[fin]
        fin=fin+1;
    

axarr[3,0].set_title('Reconstructed unblocked Image(p=100)')
axarr[3,0].imshow(img1,cmap='gray')
io.imsave("./Images/R100.png",img1)


#reconstructing image for k=1600
dempo1 = recon_img_mat2.reshape(1600,16,16)

fin = 0;
img2 = np.zeros((640,640),dtype=float)
for i in range(0,640,16):
    for j in range(0,640,16):
        img2[i:i+16, j:j+16] = dempo1[fin]
        fin=fin+1;
    

axarr[3,1].set_title('Reconstructed unblocked Image(p=1600)')
axarr[3,1].imshow(img2,cmap='gray')
io.imsave("./Images/R1600.png",img2)



# In[ ]:





# In[ ]:




