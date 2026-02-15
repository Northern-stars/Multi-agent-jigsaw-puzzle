import cv2
import numpy as np

# train_x_path = 'dataset/train_img_48gap_33-001.npy'
# train_y_path = 'dataset/train_label_48gap_33.npy'
train_x_path = 'dataset/test_img_48gap_33.npy'
train_y_path = 'dataset/test_label_48gap_33.npy'

train_x=np.load(train_x_path)
image0=train_x[1333]
train_y=np.load(train_y_path)
imageLabel=train_y[1333]
# imageLabel=list(imageLabel)
# for a in range(8):
#     for b in range(8):
#         # print(type(imageLabel[i]))
#         # print(imageLabel)
#         if imageLabel[a][b]==True:
#             if b>=4:
#                 imageLabel[a]=b+1
#                 break
#             else:
#                 imageLabel[a]=b
#                 break
imageLabel=list(np.argmax(imageLabel,axis=1))
for i in range(len(imageLabel)):
    if imageLabel[i]>=4:
        imageLabel[i]+=1
            


imageLabel.insert(4,4)
print(imageLabel)
imagefragments=[
            image0[0:96,0:96,:],
            image0[0:96,96:192,:],
            image0[0:96,192:288,:],
            image0[96:192,0:96,:],
            image0[96:192,96:192,:],
            image0[96:192,192:288,:],
            image0[192:288,0:96,:],
            image0[192:288,96:192,:],
            image0[192:288,192:288,:]
]
new_image=np.zeros([288,288,3],dtype=np.uint8)
new_image2=np.zeros([288,288,3],dtype=np.uint8)
for a in range(9):
      label_i=imageLabel[a]
    #   new_image[(0+label_i%3)*96:(1+label_i%3)*96,(0+label_i//3)*96:(1+label_i//3)*96,:]=imagefragments[i]
      new_image2[(0+label_i//3)*96:(1+label_i//3)*96,(0+label_i%3)*96:(1+label_i%3)*96,:]=imagefragments[a]
print(new_image[0][0][0])
print(type(image0[0][0][0]))
cv2.imshow("window1",image0)
# cv2.imshow("window2",new_image)
cv2.imshow("window3",new_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
