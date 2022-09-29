import cv2
import imutils

# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('vid.mp4')

while cap.isOpened():
	# Reading the video stream
	ret, image = cap.read()
	if ret:
		image = imutils.resize(image,
							width=min(400, image.shape[1]))

		# Detecting all the regions
		# in the Image that has a
		# pedestrians inside it
		(regions, _) = hog.detectMultiScale(image,
											winStride=(4, 4),
											padding=(4, 4),
											scale=1.05)

		# Drawing the regions in the
		# Image
		count=0
		for (x, y, w, h) in regions:
			cv2.rectangle(image, (x, y),
						(x + w, y + h),
						(0, 255, 0), 2)
			count+=1
		cv2.putText(image, f'Total Persons : {count}', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0,255), 2)
		# Showing the output Image
		cv2.imshow("Image", image)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
