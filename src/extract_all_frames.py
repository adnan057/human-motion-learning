import cv2
import numpy as np 
import pickle
#import cPickle as pickle

videos = pickle.load(open('data/videos_to_pickle.p','rb'))

print("Total Videos : %i" %len(videos))

sum = 0
for v in videos:
	sum += v.shape[0]

print('Total Frames : %i' %sum)

frames = []

i = 0
for video in videos:
	for frame in video:
		frames.append(frame)
		i+=1

frames = np.array(frames)

frames = frames.swapaxes(1,0)
frames = frames.swapaxes(2,1)

print(frames.shape)

cv2.imwrite('img3.png',frames[:,:,1])
pickle.dump(frames,open('data/extracted_frames.p','wb'),-1)