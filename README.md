# ComputerVision
How to run
1. Place PCA.py file and Images folder together
2. Install opencv=4.0.1 into python
3. Run .py file by using Python3 Compiler





# PCA
PCA for Image Reconsteruction

Image chosen: https://jpg2png.com/download/u09h67gnit7d4wp8/o_1en7ci4lmt3q1td2cpn1ecl1mrel/elephant-5681533_640.png?rnd=0.41940683355588937
1. The given image is read and converted into grayscale image 
   Then image is converted in 16x16 non-overlapping patches.


	Then these blocks are used to calculate PCA.
  
	•	To calculate PCA, mean of the given set of blocks is calculated
  
	•	Then with the help of mean, all the images are centred 
  
	•	Covariance matrix is calculated for the centred images
  
	•	By using Covariance, eigenvalues and eigenvectors are obtained
  
	•	Eigenvectors are used to calculate Principal Component Analysis (PCA)

	Then image is reconstructed using PCA and mean

	•	First we will sort eigenvalues descending and then we will sort eigenvectors according to eigenvalues
  
	•	Then with the help of sorted eigenvectors and mean, new image is reconstructed
  
	•	Only k amount of eigenvectors are used in reconstructing image
	
	
	Reconstructed Images are stored as R10.png and R100.png
  
	
2. Respective Images are stored as PCA1.png to PCA10.png
 


3. Selecting better value for k
   Principal Component k is basically a number of eigenvectors to be used while reconstructing an Image. 
   Eigenvectors are sorted descending so that the highest eigenvectors will be at the top.
   Then k number of highest eigenvectors is chosen while reconstructing the image.
   As we increase the number of eigenvectors to be used i.e. the value of k, we will get a more clear and accurate image.
   The choice of k depends on for what purpose we have to reconstruct the image, i.e. if we have to reduce the size of the image then we can choose the value of k such that it will provide a more clear image at minimum k.
   The best value of k is the total number of eigenvectors that will give an exact copy of an image.
   In this example, total 1600 eigenvectors are generated. Therefore, for best performance, if we set k = 1600, the we will receive reconstructed image as same as original
   
   Reconstructed image with k=1600 is stored as R1600.png
   
4. Quality of reconstructed images
   I have reconstructed an image 3 times using k as 10,100,1600.
   When k = 10, only 10 highest eigenvectors are used in reconstruction, therefore the image is not clear.
   The reconstructed 16x16 image of k=10 is also not clear when compared to the original 16x16 image.
   When k is increased to 100, we can see the image is more clear and also 16x16 patches of images are also started to become a little bit similar to the original image
   When k is set to the highest i.e. 1600, we can see, reconstructed image and also 16x16 patches of the image are an identical copy of the original image
   From this information, we can say that as we increase the value of k, the accuracy of the image also increases.
   
   