# import the necessary packages
import os

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
import cv2
import argparse
import imutils
import dlib

'''ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())'''

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/shape_predictor_68_face_landmarks.dat")
img_list = list(paths.list_images('C:/Diploma/imgDiploma/images/'))
#img_list = list(paths.list_images('C:/Diploma/imgDiploma/original_sequences/youtube/c23/images'))
out_path = "C:/Diploma/imgDiploma/manipulated_sequences_cropped/FaceSwap/"

fa = FaceAligner(predictor, desiredFaceWidth=256)
j = 0
i = 0


#for imagePath in img_list:
for i in range(j, len(img_list)):
# load the input image, resize it, and convert it to grayscale
	imagePath = img_list[i]
	print(imagePath)
	#i += 1
	image = cv2.imread(imagePath)
	#image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#cv2.imshow("Input", image)
	rects = detector(gray, 2)
	#print(len(rects))
	if len(rects) == 1:
		for rect in rects:
			try:
				(x, y, w, h) = rect_to_bb(rect)
				faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
				faceAligned = fa.align(image, gray, rect)

				f = imagePath[29:-9]
				new_name = out_path + f + '/'
				dirname = os.path.dirname(new_name)
				if not os.path.isdir(dirname):
					os.makedirs(dirname)
				p = imagePath[37:-4]
				cv2.imwrite(new_name + p + ".jpg", faceAligned)
			except cv2.error as e:
				print(str(e))
			except ZeroDivisionError:
				print("Cannot devide by zero")

			# display the output images
			'''cv2.imshow("Original", faceOrig)
			cv2.imshow("Aligned", faceAligned)
			cv2.waitKey(0)'''