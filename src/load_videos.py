import numpy as np
import cv2
#import cPickle as pickle
import pickle
import os

path = 'data/raw_data/'
num_frames = 2233	#for train
#num_frames  = 1337	#for test

action_folders = [f for f in os.listdir(path)  if not f.startswith('.')]

labels = []
frames = []
i = -1
for action in action_folders:
	video = path+action+'/'
	i+=1
	print(action + '----' + video)
	for f in os.listdir(video):
		if f.split('.')[-1] == 'avi':
			link = video+str(f)
			print(link)
			cap = cv2.VideoCapture(link)

			while True:
				ret, img = cap.read()

				if img is None:
					break

				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


				frames.append(img)
				labels.append(int(i))				

print('\nTotal frames : %i' %len(frames))
print('Total labels : %i' %len(labels))

frames = np.array(frames)
labels = np.array(labels).reshape(num_frames,1)

frames = frames.swapaxes(1,0)
frames = frames.swapaxes(2,1)

frames = frames.reshape(frames.shape[0]*frames.shape[1],num_frames)
print(frames.shape)
print(labels.shape)

pickle.dump(frames,open('data/train_data.p','wb'),-1)
pickle.dump(labels,open('data/train_labels.p','wb'),-1)

print('Pickled your files.')

