from PIL import Image
import numpy as np
import cmath
import time

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

def fft(x: np.ndarray)-> float:
    N=len(x)
    if N <= 1:
        return x
    elif N % 2 !=0:

        print("error N % 2 !=0")
    even=fft(x[0::2])
    odd=fft(x[1::2])
    temp=[cmath.exp(-2j * cmath.pi * k / N)*odd[k] for k in range(len(odd))]
    return [even[k] + temp[k] for k in range(len(odd))] + [even[k] - temp[k] for k in range(len(odd))]
#def ifft(x:np.ndarray)-> float:
#    for i in range(len(x)):
#        for j in range(len(x)):





def fft_for_DCT(x: np.ndarray)-> float:
    result= np.array([])
    x=np.pad(x,(0,len(x)),mode='symmetric')
    x=fft(x)
    for i in range(0,int(len(x)/2)):
        val=x[i]* cmath.exp(-1j * cmath.pi * i / (len(x)))
        if i==0:
            result=np.append(result,val.real*cmath.sqrt(1/(len(x)/2)))
        else:
            result=np.append(result,val.real*cmath.sqrt(2/(len(x)/2)))
    return(result)



def dct_2D(x: np.ndarray)-> np.ndarray:
    for i in range(len(x)):
        x[i]=fft_for_DCT(x[i])
    for j in range(len(x[0])):
        x[:,j]=fft_for_DCT(x[:,j])
    return x


# image = load_image("Lenna.png")
# print(image)
# image_ycbcr=(rgb_to_ycbcr(image))
# print(image_ycbcr)
# Y, Cb, Cr= sep_matrix_ycbcr(image_ycbcr)
# print(Y,Cb,Cr)

