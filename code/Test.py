from medpy.io import load, save
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import tensorflow as tf
dataPath = '../Fetal/'
modelPath = '../models/'



inputCounter = 0
NumberOfSamples = 16

windowSize0 = 25
windowSize1 = 51
windowSize2 = 15

borderCheckingSize = 5
totalNumberOfSamples = 5000

inputBatchXY0 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize0,windowSize0,2))
inputBatchXZ0 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize0,windowSize0,2))
inputBatchYZ0 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize0,windowSize0,2))

inputBatchXY1 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize1,windowSize1,2))
inputBatchXZ1 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize1,windowSize1,2))
inputBatchYZ1 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize1,windowSize1,2))

inputBatchXY2 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize2,windowSize2,2))
inputBatchXZ2 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize2,windowSize2,2))
inputBatchYZ2 = np.zeros((NumberOfSamples*totalNumberOfSamples,windowSize2,windowSize2,2))

inputBatchLabel = np.zeros(NumberOfSamples*totalNumberOfSamples)
numberOfValuesBorderChecking = borderCheckingSize * borderCheckingSize * borderCheckingSize

    # NETWORK XY
networkXY0 = input_data(shape=[None, windowSize0, windowSize0, 2], name='inputXY0')
networkXY0 = conv_2d(networkXY0, 24, 5, activation='relu',padding='same', regularizer="L2",name='XY0')
networkXY0 = max_pool_2d(networkXY0, 2)
networkXY0 = local_response_normalization(networkXY0)
networkXY0 = conv_2d(networkXY0, 32, 3, activation='relu',padding='same', regularizer="L2")
networkXY0 = max_pool_2d(networkXY0, 2)
networkXY0 = local_response_normalization(networkXY0)
networkXY0 = conv_2d(networkXY0, 48, 3, activation='relu',padding='same', regularizer="L2")
networkXY0 = max_pool_2d(networkXY0,2)
networkXY0 = fully_connected(networkXY0, 256, activation='tanh')
networkXY0 = dropout(networkXY0, 0.5)

networkXY1 = input_data(shape=[None, windowSize1, windowSize1, 2], name='inputXY1')
networkXY1 = conv_2d(networkXY1, 24, 7, activation='relu',padding='same', regularizer="L2",name='XY1')
networkXY1 = max_pool_2d(networkXY1, 2)
networkXY1 = local_response_normalization(networkXY1)
networkXY1 = conv_2d(networkXY1, 32, 5, activation='relu',padding='same', regularizer="L2")
networkXY1 = max_pool_2d(networkXY1, 2)
networkXY1 = local_response_normalization(networkXY1)
networkXY1 = conv_2d(networkXY1, 48, 3, activation='relu',padding='same', regularizer="L2")
networkXY1 = max_pool_2d(networkXY1,2)
networkXY1 = fully_connected(networkXY1, 256, activation='tanh')
networkXY1 = dropout(networkXY1, 0.5)

networkXY2 = input_data(shape=[None, windowSize2, windowSize2, 2], name='inputXY2')
networkXY2 = conv_2d(networkXY2, 24, 5, activation='relu',padding='same', regularizer="L2",name='XY2')
networkXY2 = max_pool_2d(networkXY2, 2)
networkXY2 = local_response_normalization(networkXY2)
networkXY2 = conv_2d(networkXY2, 32, 3, activation='relu',padding='same', regularizer="L2")
networkXY2 = max_pool_2d(networkXY2, 2)
networkXY2 = local_response_normalization(networkXY2)
networkXY2 = conv_2d(networkXY2, 48, 3, activation='relu',padding='same', regularizer="L2")
networkXY2 = max_pool_2d(networkXY2,2)
networkXY2 = fully_connected(networkXY2, 256, activation='tanh')
networkXY2 = dropout(networkXY2, 0.5)

conOutXY = tflearn.layers.merge_ops.merge ([networkXY0,networkXY1,networkXY2], 'concat', axis=1, name='MergeXZ')
conOutXY = fully_connected(conOutXY, 256, activation='tanh')
conOutXY = dropout(conOutXY, 0.5)

# NETWORK XZ
networkXZ0 = input_data(shape=[None, windowSize0, windowSize0, 2], name='inputXZ0')
networkXZ0 = conv_2d(networkXZ0, 24, 5, activation='relu',padding='same', regularizer="L2",name='XZ0')
networkXZ0 = max_pool_2d(networkXZ0, 2)
networkXZ0 = local_response_normalization(networkXZ0)
networkXZ0 = conv_2d(networkXZ0, 32, 3, activation='relu',padding='same', regularizer="L2")
networkXZ0 = max_pool_2d(networkXZ0, 2)
networkXZ0 = local_response_normalization(networkXZ0)
networkXZ0 = conv_2d(networkXZ0, 48, 3, activation='relu',padding='same', regularizer="L2")
networkXZ0 = max_pool_2d(networkXZ0,2)
networkXZ0 = fully_connected(networkXZ0, 256, activation='tanh')
networkXZ0 = dropout(networkXZ0, 0.5)

networkXZ1 = input_data(shape=[None, windowSize1, windowSize1, 2], name='inputXZ1')
networkXZ1 = conv_2d(networkXZ1, 24, 7, activation='relu',padding='same', regularizer="L2",name='XZ1')
networkXZ1 = max_pool_2d(networkXZ1, 2)
networkXZ1 = local_response_normalization(networkXZ1)
networkXZ1 = conv_2d(networkXZ1, 32, 5, activation='relu',padding='same', regularizer="L2")
networkXZ1 = max_pool_2d(networkXZ1, 2)
networkXZ1 = local_response_normalization(networkXZ1)
networkXZ1 = conv_2d(networkXZ1, 48, 3, activation='relu',padding='same', regularizer="L2")
networkXZ1 = max_pool_2d(networkXZ1,2)
networkXZ1 = fully_connected(networkXZ1, 256, activation='tanh')
networkXZ1 = dropout(networkXZ1, 0.5)

networkXZ2 = input_data(shape=[None, windowSize2, windowSize2, 2], name='inputXZ2')
networkXZ2 = conv_2d(networkXZ2, 24, 5, activation='relu',padding='same', regularizer="L2",name='XZ2')
networkXZ2 = max_pool_2d(networkXZ2, 2)
networkXZ2 = local_response_normalization(networkXZ2)
networkXZ2 = conv_2d(networkXZ2, 32, 3, activation='relu',padding='same', regularizer="L2")
networkXZ2 = max_pool_2d(networkXZ2, 2)
networkXZ2 = local_response_normalization(networkXZ2)
networkXZ2 = conv_2d(networkXZ2, 48, 3, activation='relu',padding='same', regularizer="L2")
networkXZ2 = max_pool_2d(networkXZ2,2)
networkXZ2 = fully_connected(networkXZ2, 256, activation='tanh')
networkXZ2 = dropout(networkXZ2, 0.5)

conOutXZ = tflearn.layers.merge_ops.merge ([networkXZ0,networkXZ1,networkXZ2], 'concat', axis=1, name='MergeXZ')
conOutXZ = fully_connected(conOutXZ, 256, activation='tanh')
conOutXZ = dropout(conOutXZ, 0.5)

# NETWORK YZ
networkYZ0 = input_data(shape=[None, windowSize0, windowSize0, 2], name='inputYZ0')
networkYZ0 = conv_2d(networkYZ0, 24, 5, activation='relu',padding='same', regularizer="L2",name='YZ0')
networkYZ0 = max_pool_2d(networkYZ0, 2)
networkYZ0 = local_response_normalization(networkYZ0)
networkYZ0 = conv_2d(networkYZ0, 32, 3, activation='relu',padding='same', regularizer="L2")
networkYZ0 = max_pool_2d(networkYZ0, 2)
networkYZ0 = local_response_normalization(networkYZ0)
networkYZ0 = conv_2d(networkYZ0, 48, 3, activation='relu',padding='same', regularizer="L2")
networkYZ0 = max_pool_2d(networkYZ0,2)
networkYZ0 = fully_connected(networkYZ0, 256, activation='tanh')
networkYZ0 = dropout(networkYZ0, 0.5)

networkYZ1 = input_data(shape=[None, windowSize1, windowSize1, 2], name='inputYZ1')
networkYZ1 = conv_2d(networkYZ1, 24, 7, activation='relu',padding='same', regularizer="L2",name='YZ1')
networkYZ1 = max_pool_2d(networkYZ1, 2)
networkYZ1 = local_response_normalization(networkYZ1)
networkYZ1 = conv_2d(networkYZ1, 32, 5, activation='relu',padding='same', regularizer="L2")
networkYZ1 = max_pool_2d(networkYZ1, 2)
networkYZ1 = local_response_normalization(networkYZ1)
networkYZ1 = conv_2d(networkYZ1, 48, 3, activation='relu',padding='same', regularizer="L2")
networkYZ1 = max_pool_2d(networkYZ1,2)
networkYZ1 = fully_connected(networkYZ1, 256, activation='tanh')
networkYZ1 = dropout(networkYZ1, 0.5)

networkYZ2 = input_data(shape=[None, windowSize2, windowSize2, 2], name='inputYZ2')
networkYZ2 = conv_2d(networkYZ2, 24, 5, activation='relu',padding='same', regularizer="L2",name='YZ2')
networkYZ2 = max_pool_2d(networkYZ2, 2)
networkYZ2 = local_response_normalization(networkYZ2)
networkYZ2 = conv_2d(networkYZ2, 32, 3, activation='relu',padding='same', regularizer="L2")
networkYZ2 = max_pool_2d(networkYZ2, 2)
networkYZ2 = local_response_normalization(networkYZ2)
networkYZ2 = conv_2d(networkYZ2, 48, 3, activation='relu',padding='same', regularizer="L2")
networkYZ2 = max_pool_2d(networkYZ2,2)
networkYZ2 = fully_connected(networkYZ2, 256, activation='tanh')
networkYZ2 = dropout(networkYZ2, 0.5)

conOutYZ = tflearn.layers.merge_ops.merge ([networkYZ0,networkYZ1,networkYZ2], 'concat', axis=1, name='MergeYZ')
conOutYZ = fully_connected(conOutYZ, 256, activation='tanh')
conOutYZ = dropout(conOutYZ, 0.5)

#MERGING
conOutALL = tflearn.layers.merge_ops.merge ([conOutXY,conOutXZ,conOutYZ], 'concat', axis=1, name='Merge')
conOutALL = fully_connected(conOutALL, 2, activation='softmax')
conOutALL = regression(conOutALL, optimizer='adam', learning_rate=0.0001,
                     loss='categorical_crossentropy', name='target')
model = tflearn.DNN(conOutALL, tensorboard_verbose=0)


for step in xrange(2):
    model.load(modelPath+'AutoContext'+str(step)+'.tflearn')
    from scipy import stats

    for f in listdir(dataPath+'Final_Subjects_Auto/'):
        if "nii" in f:
            print f
            image_data, image_header = load(dataPath+'Final_Subjects_Auto/'+f)
#             image_data = image_data/np.linalg.norm(image_data)
            image_data_labels, image_header_labels = load(dataPath+'AllLabels/'+'mask_'+f[1:])
            image_data_labels = np.clip(image_data_labels, 0, 1)
            image_data_labels = image_data_labels.astype(np.float64)
    #         TempPrimaryLable = np.load(dataPath+'Results/'+f[:7]+'_predictedprob_LPBA_9parallel_AWS_small_2_res.npy')
            if step > 0:
                posterior = np.load(dataPath+'G5posterior/'+str(step-1)+f[:-4]+'.npy')
            else:
                posterior = np.zeros_like(image_data_labels) + 0.5

    #         PrimaryLable = TempPrimaryLable[:,:,:,1]
    #         PrimaryLable = stats.threshold(PrimaryLable, threshmin=0.9, threshmax=1, newval=0)
    #         PrimaryLable = stats.threshold(PrimaryLable, threshmin=0, threshmax=0.9, newval=1)
    #         print 'before secondary:'
    #         tp = np.sum(np.multiply(PrimaryLable,image_data_labels))
    #         tn = np.sum(np.multiply((1-PrimaryLable),(1-image_data_labels)))
    #         fp = np.sum(np.multiply(PrimaryLable,(1-image_data_labels)))
    #         fn = np.sum(np.multiply((1-PrimaryLable),image_data_labels))
    #         print('dice:')
    #         print(2*tp/(2*tp+fp+fn))
    #         print('Sensitivity:')
    #         print(tp/(tp+fn))
    #         print('Specifity:')
    #         print(tn/(tn+fp))

            PrimaryLable = posterior*128
            imageDim = np.shape(image_data)
            expandedImageData0 = np.zeros((imageDim[0]+windowSize0,imageDim[1]+windowSize0,imageDim[2]+windowSize0))
            expandedImageData0[(windowSize0-1)/2+1:-(windowSize0-1)/2,(windowSize0-1)/2+1:-(windowSize0-1)/2,
                              (windowSize0-1)/2+1:-(windowSize0-1)/2] = image_data
            expandedImageData1 = np.zeros((imageDim[0]+windowSize1,imageDim[1]+windowSize1,imageDim[2]+windowSize1))
            expandedImageData1[(windowSize1-1)/2+1:-(windowSize1-1)/2,(windowSize1-1)/2+1:-(windowSize1-1)/2,
                              (windowSize1-1)/2+1:-(windowSize1-1)/2] = image_data
            expandedImageData2 = np.zeros((imageDim[0]+windowSize2,imageDim[1]+windowSize2,imageDim[2]+windowSize2))
            expandedImageData2[(windowSize2-1)/2+1:-(windowSize2-1)/2,(windowSize2-1)/2+1:-(windowSize2-1)/2,
                              (windowSize2-1)/2+1:-(windowSize2-1)/2] = image_data

            expandedImageLabel0 = np.zeros((imageDim[0]+windowSize0,imageDim[1]+windowSize0,imageDim[2]+windowSize0))
            expandedImageLabel0[(windowSize0-1)/2+1:-(windowSize0-1)/2,(windowSize0-1)/2+1:-(windowSize0-1)/2,
                              (windowSize0-1)/2+1:-(windowSize0-1)/2] = PrimaryLable
            expandedImageLabel1 = np.zeros((imageDim[0]+windowSize1,imageDim[1]+windowSize1,imageDim[2]+windowSize1))
            expandedImageLabel1[(windowSize1-1)/2+1:-(windowSize1-1)/2,(windowSize1-1)/2+1:-(windowSize1-1)/2,
                              (windowSize1-1)/2+1:-(windowSize1-1)/2] = PrimaryLable
            expandedImageLabel2 = np.zeros((imageDim[0]+windowSize2,imageDim[1]+windowSize2,imageDim[2]+windowSize2))
            expandedImageLabel2[(windowSize2-1)/2+1:-(windowSize2-1)/2,(windowSize2-1)/2+1:-(windowSize2-1)/2,
                              (windowSize2-1)/2+1:-(windowSize2-1)/2] = PrimaryLable

            expandedImageLabel = np.zeros((imageDim[0]+windowSize0,imageDim[1]+windowSize0,imageDim[2]+windowSize0))
            expandedImageLabel[(windowSize0-1)/2+1:-(windowSize0-1)/2,(windowSize0-1)/2+1:-(windowSize0-1)/2,
                              (windowSize0-1)/2+1:-(windowSize0-1)/2] = image_data_labels
            predicted_label = np.zeros_like(image_data)
            predicted_prob = np.zeros((imageDim[0],imageDim[1],imageDim[2],2))

            pixelWindowsXY0 = np.zeros((imageDim[2],windowSize0,windowSize0,2))
            pixelWindowsXZ0 = np.zeros((imageDim[2],windowSize0,windowSize0,2))
            pixelWindowsYZ0 = np.zeros((imageDim[2],windowSize0,windowSize0,2))

            pixelWindowsXY1 = np.zeros((imageDim[2],windowSize1,windowSize1,2))
            pixelWindowsXZ1 = np.zeros((imageDim[2],windowSize1,windowSize1,2))
            pixelWindowsYZ1 = np.zeros((imageDim[2],windowSize1,windowSize1,2))

            pixelWindowsXY2 = np.zeros((imageDim[2],windowSize2,windowSize2,2))
            pixelWindowsXZ2 = np.zeros((imageDim[2],windowSize2,windowSize2,2))
            pixelWindowsYZ2 = np.zeros((imageDim[2],windowSize2,windowSize2,2))

            pixelCounter = 0
            for x in xrange(0,imageDim[0]):
                for y in xrange(0,imageDim[1]):
                    pixelCounter = 0
                    for z in xrange(0,imageDim[2]):
                        pixelWindowsXY0[pixelCounter,:,:,0] = expandedImageData0[x+1:x+(windowSize0-1)+2,y+1:y+(windowSize0-1)+2,
                                                                                z+(windowSize0-1)/2+1]
                        pixelWindowsXZ0[pixelCounter,:,:,0] = expandedImageData0[x+1:x+(windowSize0-1)+2,y+(windowSize0-1)/2+1,
                                                                                z+1:z+(windowSize0-1)+2]
                        pixelWindowsYZ0[pixelCounter,:,:,0] = expandedImageData0[x+(windowSize0-1)/2+1,y+1:y+(windowSize0-1)+2,
                                                                                z+1:z+(windowSize0-1)+2]

                        pixelWindowsXY1[pixelCounter,:,:,0] = expandedImageData1[x+1:x+(windowSize1-1)+2,y+1:y+(windowSize1-1)+2,
                                                                                z+(windowSize1-1)/2+1]
                        pixelWindowsXZ1[pixelCounter,:,:,0] = expandedImageData1[x+1:x+(windowSize1-1)+2,y+(windowSize1-1)/2+1,
                                                                                z+1:z+(windowSize1-1)+2]
                        pixelWindowsYZ1[pixelCounter,:,:,0] = expandedImageData1[x+(windowSize1-1)/2+1,y+1:y+(windowSize1-1)+2,
                                                                                z+1:z+(windowSize1-1)+2]

                        pixelWindowsXY2[pixelCounter,:,:,0] = expandedImageData2[x+1:x+(windowSize2-1)+2,y+1:y+(windowSize2-1)+2,
                                                                                z+(windowSize2-1)/2+1]
                        pixelWindowsXZ2[pixelCounter,:,:,0] = expandedImageData2[x+1:x+(windowSize2-1)+2,y+(windowSize2-1)/2+1,
                                                                                z+1:z+(windowSize2-1)+2]
                        pixelWindowsYZ2[pixelCounter,:,:,0] = expandedImageData2[x+(windowSize2-1)/2+1,y+1:y+(windowSize2-1)+2,
                                                                                z+1:z+(windowSize2-1)+2]

                        pixelWindowsXY0[pixelCounter,:,:,1] = expandedImageLabel0[x+1:x+(windowSize0-1)+2,y+1:y+(windowSize0-1)+2,
                                                                                z+(windowSize0-1)/2+1]
                        pixelWindowsXZ0[pixelCounter,:,:,1] = expandedImageLabel0[x+1:x+(windowSize0-1)+2,y+(windowSize0-1)/2+1,
                                                                                z+1:z+(windowSize0-1)+2]
                        pixelWindowsYZ0[pixelCounter,:,:,1] = expandedImageLabel0[x+(windowSize0-1)/2+1,y+1:y+(windowSize0-1)+2,
                                                                                z+1:z+(windowSize0-1)+2]

                        pixelWindowsXY1[pixelCounter,:,:,1] = expandedImageLabel1[x+1:x+(windowSize1-1)+2,y+1:y+(windowSize1-1)+2,
                                                                                z+(windowSize1-1)/2+1]
                        pixelWindowsXZ1[pixelCounter,:,:,1] = expandedImageLabel1[x+1:x+(windowSize1-1)+2,y+(windowSize1-1)/2+1,
                                                                                z+1:z+(windowSize1-1)+2]
                        pixelWindowsYZ1[pixelCounter,:,:,1] = expandedImageLabel1[x+(windowSize1-1)/2+1,y+1:y+(windowSize1-1)+2,
                                                                                z+1:z+(windowSize1-1)+2]

                        pixelWindowsXY2[pixelCounter,:,:,1] = expandedImageLabel2[x+1:x+(windowSize2-1)+2,y+1:y+(windowSize2-1)+2,
                                                                                z+(windowSize2-1)/2+1]
                        pixelWindowsXZ2[pixelCounter,:,:,1] = expandedImageLabel2[x+1:x+(windowSize2-1)+2,y+(windowSize2-1)/2+1,
                                                                                z+1:z+(windowSize2-1)+2]
                        pixelWindowsYZ2[pixelCounter,:,:,1] = expandedImageLabel2[x+(windowSize2-1)/2+1,y+1:y+(windowSize2-1)+2,
                                                                                z+1:z+(windowSize2-1)+2]
                        pixelCounter += 1
            #             pixelWindows = pixelWindows.reshape([-1, windowSize0, windowSize0, 1])
                    predicted = model.predict([pixelWindowsXY0,pixelWindowsXY1,pixelWindowsXY2,
                                                                      pixelWindowsXZ0,pixelWindowsXZ1,pixelWindowsXZ2,
                                                                      pixelWindowsYZ0,pixelWindowsYZ1,pixelWindowsYZ2])
                    predicted_label[x,y,:] = np.argmax(predicted,axis=1)
                    predicted_prob[x,y,:,:] = predicted
                print x
            np.save(dataPath+'G5posterior/'+str(step)+f[:-4],predicted_prob[:,:,:,1])
            predicted_Thrsh = np.zeros_like(image_data_labels)
            predicted_Thrsh = predicted_prob[:,:,:,1]
            predicted_Thrsh = stats.threshold(predicted_Thrsh, threshmin=0.5, threshmax=1, newval=0)
            predicted_Thrsh = stats.threshold(predicted_Thrsh, threshmin=0, threshmax=0.5, newval=1)
            tp = np.sum(np.multiply(predicted_Thrsh,image_data_labels))
            tn = np.sum(np.multiply((1-predicted_Thrsh),(1-image_data_labels)))
            fp = np.sum(np.multiply(predicted_Thrsh,(1-image_data_labels)))
            fn = np.sum(np.multiply((1-predicted_Thrsh),image_data_labels))
            print('dice:')
            print(2*tp/(2*tp+fp+fn))
            print('Sensitivity:')
            print(tp/(tp+fn))
            print('Specifity:')
            print(tn/(tn+fp))