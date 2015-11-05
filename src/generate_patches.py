import pickle
import random
#import cPickle as pickle
import numpy as np
import cv2

frames = pickle.load(open('data/extracted_frames.p','rb'))

print("Total Videos : %i" %len(frames))
num_patches = 100000

def get_patches(videos):

	patch_size = 15
	image_rows = 144
	image_cols = 180

	patches = np.zeros((15*15,num_patches),dtype=np.float64)

	print(patches.shape)
   
	for i in range(num_patches):

		frame_num = int(random.uniform(0,len(frames)))
		
		print("Reading : %i" %(i+1))

		x = random.randint(0,image_rows - patch_size)
		y = random.randint(0,image_cols - patch_size)

		patch = frames[x:x + patch_size, y:y + patch_size,frame_num].reshape(patch_size*patch_size)
		patch = patch.flatten()

		patches[:,i] = patch
	
	return patches

patches = get_patches(frames)
#print(patches[:,:10])   #get first 10 patches

pickle.dump(patches,open('data/patches_'+str(num_patches)+'.p','wb'),-1)	
print('\nTotal, ' + str(num_patches) + ' patches extracted...')

