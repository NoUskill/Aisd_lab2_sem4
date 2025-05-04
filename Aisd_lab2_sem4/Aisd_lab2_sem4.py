from PIL import Image
import numpy as np
import cmath
import time
from scipy.fft import dct, idct, dctn
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

def fft(x: np.ndarray):
    N=len(x)
    if N <= 1:
        return x
    elif N % 2 !=0:
        print("error N % 2 !=0 in fft")
    even=fft(x[0::2])
    odd=fft(x[1::2])
    temp=[cmath.exp(-2j * cmath.pi * k / N)*odd[k] for k in range(len(odd))]
    return np.array([even[k] + temp[k] for k in range(len(odd))] + [even[k] - temp[k] for k in range(len(odd))])





    return x

def fft_for_DCT(x: np.ndarray):
    result= np.array([])
    x_temp=np.pad(x,(0,len(x)),mode='symmetric')
    x_fft=fft(x_temp)
    N=int(len(x_fft)/2)
    result = np.zeros(N, dtype=np.float64)

    for i in range(N):
        val = x_fft[i] * cmath.exp(-1j * cmath.pi * i / (2 * N))
        if i == 0:
            result[i] = (val * np.sqrt(1/N)).real*2*np.sqrt(2)
        else:
            result[i] = (val * np.sqrt(2/N)).real*2

    return(result)



def dct_2D(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for i in range(len(x)):
        x[i] = fft_for_DCT(x[i])
    for j in range(len(x[0])):
        x[:, j] = fft_for_DCT(x[:, j])
    return x

def idct_2D(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for i in range(len(x)):
        x[i] = my_idct(x[i])
    for j in range(len(x[0])):
        x[:,j] = my_idct(x[:,j])
    return x


# x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
# x = np.array([4, 2, 2, 2, 0, 6, 7, 4], dtype=np.float64)
# x = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
# print("Your Arr:",x)
# x_fft_for_DCT=fft_for_DCT(x)
# print("Your DCT:",x_fft_for_DCT)
# x_copy=ifft_for_IDCT(x_fft_for_DCT)
# t_2=idct(x_fft_for_DCT)
# print("Your IDCT:", x_copy)
# print("SciPy IDCT:", t_2)
# print("Ratios:",t_2/ x_copy)



x_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8])


