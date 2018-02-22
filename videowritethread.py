import threading
import cv2
class videowritethread(threading.Thread):
	def __init__(self, threadID, name, filename, filecontent):
		threading.Thread.__init__(self)
		self.fname = filename
		self.threadID = threadID
		self.name = name
		self.content = filecontent
	def run(self):
		cv2.imwrite('{}.jpg'.format(self.fname), self.content)
