import cv2
from threading import Thread
import imutils
from collections import deque
import time


# import cv2
# import threading
# import imutils

# class camThread(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.sources = []
#         self.capture = None
#     def run(self):
#         for camNumber in self.sources:
#             print("Starting Camera" + camNumber)
#             # camPreview("Camera " + camNumber, camNumber)
#             cv2.namedWindow("Camera" + camNumber)
#             cam = cv2.VideoCapture(camNumber)
#             if cam.isOpened():
#                 ret, frame = cam.read()
#             else:
#                 rval=False
#             self.capture = cv2.VideoCapture(self.camera_stream_link)
                
#     def checkCamsource(self, totalCam):
#         for i in range(0, totalCam):
#             cap = cv2.VideoCapture(i)
#             if cap.read()[0]:
#                 self.sources.append(i)
#             cap.release()
#         return self.sources

            

# def camPreview(previewName, camID):
#     cv2.namedWindow(previewName)
#     cam = cv2.VideoCapture(camID)
#     if cam.isOpened():  # try to get the first frame
#         rval, frame = cam.read()
#     else:
#         rval = False

#     while rval:
#         # cv2.imshow(previewName, frame)
#         rval, frame = cam.read()
#         key = cv2.waitKey(20)
#         if key == 27:  # exit on ESC
#             break
#     cv2.destroyWindow(previewName)

# # Create two threads as follows
# thread1 = camThread("Camera 1", 1)
# thread2 = camThread("Camera 2", 2)
#------------------------------------------------------------------------------

# class VideoCamera(object):
# 	def __init__(self, camId, stream_link=0, deque_size=1, ):
# 		self.deque = deque(maxlen=deque_size)
# 		self.capture = None
# 		self.name = 'Camera_' + str(camId)
# 		self.video_frame = None
# 		self.get_frame_thread = Thread(target=self.get_frame, args=())
# 		self.get_frame_thread.daemon = True
# 		self.get_frame_thread.start()
# 		self.camera_stream_link = stream_link

# 	def load_network_stream(self):
# 		"""Verifies stream link and open new stream if valid"""
# 		def load_network_stream_thread():
# 			if self.verify_network_stream(self.camera_stream_link):
# 				self.capture = cv2.VideoCapture(self.camera_stream_link)

# 		self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
# 		self.load_stream_thread.daemon = True
# 		self.load_stream_thread.start()

# 	def verify_network_stream(self, id):
# 		cap = cv2.VideoCapture(id)
# 		if not cap.isOpened():
# 			return False
# 		cap.release()
# 		return True

# 	def get_frame(self):
# 		"""Reads frame, resizes, and converts image to pixmap"""

# 		while True:
# 			try:
# 				if self.capture.isOpened():
# 					# Read next frame from stream and insert into 	
# 					status, frame = self.capture.read()
# 					if status:
# 						self.deque.append(frame)
# 					else:
# 						self.capture.release()
# 				# else:
# 				# 	# Attempt to reconnect
# 				# 	print('attempting to reconnect', self.camera_stream_link)
# 				# 	self.load_network_stream()
# 				# 	self.spin(2)
# 				# self.spin(.001)
# 					time.sleep(0.001)
# 			except AttributeError:
# 				pass

# 	def get_frame(self):
# 		if self.deque and self.online:
# 			# Grab latest frame
# 			frame = self.deque[-1]
# 			frame_flip = cv2.flip(frame,1)
# 			ret, jpeg = cv2.imencode('.jpg', frame_flip)
# 			self.video_frame = jpeg.tobytes()

# 	def get_video_frame(self):
# 		self.get_frame()
# 		return self.video_frame
	#-----------------------------------------------------------------------------------------------------
class VideoCamera(object):
	def __init__(self, camID):
		self.video = cv2.VideoCapture(camID)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		frame_flip = cv2.flip(image,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()