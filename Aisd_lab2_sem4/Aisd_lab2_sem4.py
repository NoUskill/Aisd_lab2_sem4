from PIL import Image
import numpy as np

def load_image(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)
def rgb_to_ycbcr(image_rgb):
     transform = np.array([
        [ 0.299,     0.587,     0.114],
        [-0.168736, -0.331264,  0.5],
        [ 0.5,      -0.418688, -0.081312]
        ])
     shift = np.array([0, 128, 128])
     image_ycbcr=np.round(image_rgb @ transform.T + shift,1) 
     return image_ycbcr
def sep_matrix_ycbcr(image_ycbcr):
    Y=image_ycbcr[...,0]
    Cb=image_ycbcr[...,1]
    Cr=image_ycbcr[...,2]
    return Y, Cb, Cr
def downsample(koef:int, Matrix):
    w,h = Matrix.shape
    if h % koef != 0:
        Matrix=np.pad(Matrix,((0,koef-(h % koef)),(0,0)),mode='constant', constant_values=0)
    if w % koef !=0:
        Matrix=np.pad(Matrix,((0,0),(0,koef-(w % koef))),mode='constant', constant_values=0)
    w=(w+w % koef) / koef
    h=(h+h % koef) / koef
    print(Matrix)
    if h==w==1:
        downsample_matrix = np.array([[np.mean(Matrix[0*koef:(0*koef)+koef,0*koef:(0*koef)+koef])]])
        return downsample_matrix
    else:
        downsample_matrix=np.zeros((int(h),int(w)))
        for i in range(0, int(w)):
            for j in range(0,int(h)):
                downsample_matrix[i][j]=np.mean(Matrix[i*koef:(i*koef)+koef,j*koef:(j*koef)+koef])
        return downsample_matrix
   
def split_block(n: int, Matrix):
    w,h = Matrix.shape
    if h % n != 0:
        Matrix=np.pad(Matrix,((0,n-(h % n)),(0,0)),mode='constant', constant_values=0)
    if w % n !=0:
        Matrix=np.pad(Matrix,((0,0),(0,n-(w % n))),mode='constant', constant_values=0)

    w,h = Matrix.shape
    print(Matrix)
    split_matrix=Matrix.reshape(h//n, n, w//n, n).transpose(0, 2, 1, 3)

    return(split_matrix)




        




# image = load_image("Lenna.png")
# print(image)
# image_ycbcr=(rgb_to_ycbcr(image))
# print(image_ycbcr)
# Y, Cb, Cr= sep_matrix_ycbcr(image_ycbcr)
# print(Y,Cb,Cr)


a =  np.random.randint(0, 10, (10, 10))
print(a)
print(downsample(6,a))
temp=split_block(8,a)
