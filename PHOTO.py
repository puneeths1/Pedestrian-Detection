import cv2
import imutils

# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Reading the Image
image = cv2.imread('in.jpg')

count=0
# Resizing the Image
ima = imutils.resize(image,
					width=min(400, image.shape[1]))


# Detecting all the regions in the
# Image that has a pedestrians inside it
(regions, _) = hog.detectMultiScale(ima,
									winStride=(4, 4),
									padding=(4, 4),
									scale=1.05)
count=0
# Drawing the regions in the Image
for (x, y, w, h) in regions:
	cv2.rectangle(ima, (x, y),
				(x + w, y + h),
				(0, 255, 0), 2)	
	count+=1
image1=cv2.resize(image,(400,260))
cv2.imshow("Input Image",image1)
# Showing the output Image
cv2.putText(ima, f'Total Persons : {count}', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0,255),2)
cv2.imshow("Output Image", ima)
cv2.waitKey(0)

cv2.destroyAllWindows()
