# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import cmath
import math
import heapq
import time
import json
import struct
from scipy.fft import ifft,dct,idct, dctn,idctn
class JPEGCompressor:
    def __init__(self):
      
        pass
    def load_image(self, path):

        image = Image.open(path).convert("RGB")
        return np.array(image)
    def rgb_to_ycbcr(self, image_rgb):

        transform = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ])
        shift = np.array([0, 128, 128])
        image_ycbcr = np.round(image_rgb @ transform.T + shift, 1)
        print(f"YCbCr: min={image_ycbcr.min()}, max={image_ycbcr.max()}, shape={image_ycbcr.shape}")
        return image_ycbcr
    def sep_matrix_ycbcr(self, image_ycbcr):

        Y = image_ycbcr[..., 0]
        Cb = image_ycbcr[..., 1]
        Cr = image_ycbcr[..., 2]
        print(f"Y shape={Y.shape}, Cb shape={Cb.shape}, Cr shape={Cr.shape}")
        return Y, Cb, Cr
    def downsample(self, koef: int, Matrix):

        w, h = Matrix.shape
        if h % koef != 0:
            Matrix = np.pad(Matrix, ((0, koef - (h % koef)), (0, 0)), mode='constant', constant_values=128)
        if w % koef != 0:
            Matrix = np.pad(Matrix, ((0, 0), (0, koef - (w % koef))), mode='constant', constant_values=128)
        w, h = Matrix.shape
        w = w // koef
        h = h // koef

        if h == w == 1:
            downsample_matrix = np.array([[np.mean(Matrix[0*koef:(0*koef)+koef, 0*koef:(0*koef)+koef])]])
        else:
            downsample_matrix = np.zeros((int(h), int(w)))
            for i in range(0, int(w)):
                for j in range(0, int(h)):
                    downsample_matrix[i,j] = np.mean(Matrix[i*koef:(i*koef)+koef, j*koef:(j*koef)+koef])
        print(f"Downsampled shape={downsample_matrix.shape}, min={downsample_matrix.min()}, max={downsample_matrix.max()}")
        return downsample_matrix
    def split_block(self, n: int, Matrix):

        w, h = Matrix.shape
        if h % n != 0:
            Matrix = np.pad(Matrix, ((0, n - (h % n)), (0, 0)), mode='constant', constant_values=0)
        if w % n != 0:
            Matrix = np.pad(Matrix, ((0, 0), (0, n - (w % n))), mode='constant', constant_values=0)
        w, h = Matrix.shape
        split_matrix = Matrix.reshape(h//n, n, w//n, n).transpose(0, 2, 1, 3)
        print(f"Split blocks shape={split_matrix.shape}")
        return split_matrix
    def matrix_dct(self,Matrix):
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
        def fft_for_DCT(x: np.ndarray):
            result= np.array([])
            x_temp=np.pad(x,(0,len(x)),mode='symmetric')

            x_fft=fft(x_temp)
            N=int(len(x_fft)/2)
            result = np.zeros(N, dtype=np.float64)

            for i in range(N):
                val = x_fft[i] * cmath.exp(-1j * cmath.pi * i / (2 * N))
                if i == 0:
                    result[i] = round((val * np.sqrt(1/N)).real*2*np.sqrt(2),2)
                else:
                    result[i] = round((val * np.sqrt(2/N)).real*2, 4)

            return(result)
        def dct_2D(x: np.ndarray) -> np.ndarray:
            x = x.copy()
            for i in range(len(x)):
                x[i]=dct(x[i])
                # x[i] = fft_for_DCT(x[i])
            for j in range(len(x[0])):
                x[:, j]=dct(x[:, j])
                # x[:, j] = fft_for_DCT(x[:, j])
            return x
        N=len(Matrix)
        for i in range(N):
            for j in range(N):
                Matrix[i][j]=dctn(Matrix[i][j])
        return Matrix
    def matrix_idct(self, Matrix):
        def ifft_for_IDCT(x: np.ndarray):
            c = x.copy()
            N = len(c)
            c[0] *= np.sqrt(2)
            x_temp = np.pad(c, (0, len(c)), mode='symmetric')
            x_ifft = ifft(x_temp)
            for i in range(len(x_ifft)):
                x_ifft[i] = x_ifft[i] / len(x_ifft)
            x = np.zeros(N)
            for k in range(N):
                val = x_ifft[k] * cmath.exp(+1j * cmath.pi * k / (2 * N)) * 2
                x[k] = (val * np.sqrt(N) / (4 * np.sqrt(2))).real
            return x
        def idct_2D(c: np.ndarray) -> np.ndarray:
            x = c.copy()
            for i in range(len(x)):
                x[i] = idct(x[i])
            for j in range(len(x[0])):
                x[:, j] = idct(x[:, j])
            return x

        N1, N2 = Matrix.shape[:2]
        result = np.zeros_like(Matrix)
        for i in range(N1):
            for j in range(N2):
                result[i, j] = idctn(Matrix[i, j])
        print(f"IDCT shape={result.shape}, min={result.min()}, max={result.max()}")
        return result
    def Quant(self, Matrix, mode: str, quality: int = 50):
        if quality<10:
            quality=10
        elif quality>100:
            quality=100
        # Более агрессивное масштабирование для низкого качества
        scale = (100 / quality) ** 2.5 if quality < 50 else 50 / quality if quality < 100 else 1
        # Порог для обнуления AC-коэффициентов (более высокий для Cb/Cr)
        threshold = max(1, (100 - quality) / 1) if quality < 80 else 0
        if mode in ["Cb", "Cr"]:
            threshold *= 1.5  # Усиливаем обнуление для цветовых компонент

        if mode == "Y":
            LUMINANCE_QUANT_TABLE = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ]) * scale
            N1, N2 = Matrix.shape[:2]
            result = np.zeros_like(Matrix)
            for i in range(N1):
                for j in range(N2):
                    result[i, j] = np.round(Matrix[i, j] / LUMINANCE_QUANT_TABLE).astype(np.int32)
                    # Обнуляем малые AC-коэффициенты (кроме DC [0,0])
                    result[i, j, 1:, 1:][np.abs(result[i, j, 1:, 1:]) < threshold] = 0

            non_zero_dc = np.sum(np.abs(result[:, :, 0, 0]) > 0)
            non_zero_ac = np.sum(np.abs(result[:, :, 1:, 1:]) > 0)
            print(f"Quant Y min={result.min()}, max={result.max()}, non-zero DC={non_zero_dc}, non-zero AC={non_zero_ac}")
            return result
        elif mode in ["Cb", "Cr"]:
            CHROMINANCE_QUANT_TABLE = np.array([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ]) * scale
            N1, N2 = Matrix.shape[:2]
            result = np.zeros_like(Matrix)
            for i in range(N1):
                for j in range(N2):
                    result[i, j] = np.round(Matrix[i, j] / CHROMINANCE_QUANT_TABLE).astype(np.int32)
                    # Обнуляем малые AC-коэффициенты (кроме DC [0,0])
                    result[i, j, 1:, 1:][np.abs(result[i, j, 1:, 1:]) < threshold] = 0
            non_zero_dc = np.sum(np.abs(result[:, :, 0, 0]) > 0)
            non_zero_ac = np.sum(np.abs(result[:, :, 1:, 1:]) > 0)
            print(f"Quant {mode} min={result.min()}, max={result.max()}, non-zero DC={non_zero_dc}, non-zero AC={non_zero_ac}")
            return result
        else:
            raise ValueError("eroro")

    def iQuant(self, Matrix, mode: str, quality: int = 50):
        
        if quality<10:
            quality=10
        elif quality>100:
            quality=100
        scale = (100 / quality) ** 2.5 if quality < 50 else 50 / quality if quality < 100 else 1
        if mode == "Y":
            LUMINANCE_QUANT_TABLE = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ]) * scale
            N1, N2 = Matrix.shape[:2]
            result = np.zeros_like(Matrix)
            for i in range(N1):
                for j in range(N2):
                    result[i, j] = np.round(Matrix[i, j] * LUMINANCE_QUANT_TABLE).astype(np.int32)
            print(f"iQuant Y min={result.min()}, max={result.max()}")
            return result
        elif mode in ["Cb", "Cr"]:
            CHROMINANCE_QUANT_TABLE = np.array([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ]) * scale
            N1, N2 = Matrix.shape[:2]
            result = np.zeros_like(Matrix)
            for i in range(N1):
                for j in range(N2):
                    result[i, j] = np.round(Matrix[i, j] * CHROMINANCE_QUANT_TABLE).astype(np.int32)
            print(f"iQuant {mode} min={result.min()}, max={result.max()}")
            return result
        else:
            raise ValueError("eror")

    def split_DC_and_AC(self, Matrix):
        def zigzag_scan(matrix, size=8):
            result = []
            i, j = 0, 0
            going_up = True
            while len(result) < size * size:
                result.append(matrix[i, j])
                if going_up:
                    if i == 0 and j < size - 1:
                        j += 1
                        going_up = False
                    elif j == size - 1:
                        i += 1
                        going_up = False
                    else:
                        i -= 1
                        j += 1
                else:
                    if j == 0 and i < size - 1:
                        i += 1
                        going_up = True
                    elif i == size - 1:
                        j += 1
                        going_up = True
                    else:
                        i += 1
                        j -= 1
            return np.array(result)[1:]

        N = len(Matrix)
        list_DC = []
        temp_num = 0
        list_AC = np.array([])
        for i in range(N):
            for j in range(N):
                if i == j == 0:
                    temp_num = int((Matrix[i][j])[0][0])
                    list_DC.append(int((Matrix[i][j])[0][0]))
                    list_AC = zigzag_scan(Matrix[i][j], 8)
                else:
                    list_DC.append(int(Matrix[i][j][0][0]) - temp_num)
                    temp_num = int(Matrix[i][j][0][0])
                    list_AC = np.concatenate((list_AC, zigzag_scan(Matrix[i][j])))
        print(f"DC len={len(list_DC)}, AC len={len(list_AC)}")
        return np.array(list_DC), np.array(list_AC)
    def cod_Huff_DC(self, arr_DC):
        class Huffman_root:
            def __init__(self, sim: str, freq: int):
                self.sim = sim
                self.freq = freq
                self.leftchild = None
                self.rightchild = None
            def __lt__(self, other):
                return self.freq < other.freq
            def __str__(self):
                return self.sim + ' ' + str(self.freq)

        def assem_huf_table(tables_huf_cod: dict, root: Huffman_root, link: str):
            if root.sim is None:
                assem_huf_table(tables_huf_cod, root.leftchild, link=link + '0')
                assem_huf_table(tables_huf_cod, root.rightchild, link=link + '1')
            else:
                tables_huf_cod[root.sim] = link
            return tables_huf_cod

        arr_len = []
        for i in range(len(arr_DC)):
            if arr_DC[i] == 0:
                arr_len.append(-1)
            elif abs(arr_DC[i]) == 1:
                arr_len.append(0)
            elif abs(arr_DC[i]) == 2:
                arr_len.append(1)
            else:
                num = math.ceil(math.log(abs(arr_DC[i]), 2))
                arr_len.append(num)

        tables_freq = {}
        for i in arr_len:
            if i in tables_freq:
                tables_freq[i] = tables_freq[i] + 1
            else:
                tables_freq[i] = 1

        queue = [Huffman_root(str(key), tables_freq[key]) for key in tables_freq.keys()]
        heapq.heapify(queue)
        while len(queue) > 1:
            temp_left = heapq.heappop(queue)
            temp_right = heapq.heappop(queue)
            mid = Huffman_root(None, temp_left.freq + temp_right.freq)
            mid.leftchild = temp_left
            mid.rightchild = temp_right
            heapq.heappush(queue, mid)

        tables_huf_cod = {}
        huf_table = assem_huf_table(tables_huf_cod, queue[0], "")
        bit_huf_cod = ""
        for i in range(len(arr_DC)):
            if arr_DC[i] == 0:
                num = -1
            elif abs(arr_DC[i]) == 1:
                num = 0
            elif abs(arr_DC[i]) == 2:
                num = 1
            else:
                num = len(bin(abs(arr_DC[i]))[2:])
            bit_huf_cod += huf_table[str(num)]
            if num == -1:
                bit_huf_cod += '0'
            elif num == 0:
                if arr_DC[i] == -1:
                    bit_huf_cod += "00"
                else:
                    bit_huf_cod += "01"
            elif num == 1:
                if arr_DC[i] == -2:
                    bit_huf_cod += "01"
                else:
                    bit_huf_cod += "10"
            elif arr_DC[i] < 0:
                bit_huf_cod += ''.join('1' if bit == '0' else '0' for bit in bin(abs(arr_DC[i]))[2:])
            else:
                bit_huf_cod += bin(abs(arr_DC[i]))[2:]
        print(f"DC Huffman code length={len(bit_huf_cod)}")
        return bit_huf_cod, huf_table
    def decod_Huff_DC(self, bit_huf_cod, huf_table):
        i = 0
        temp_cod = ""
        decod_arr = []
        reversed_dict = {v: k for k, v in huf_table.items()}
        while i < len(bit_huf_cod) - 1:
            temp_cod += bit_huf_cod[i]
            if reversed_dict.get(temp_cod) is not None:
                klass = int(reversed_dict[temp_cod])
                if klass == -1:
                    decod_arr.append(0)
                    temp_cod = ''
                    i += 2
                elif klass == 0:
                    temp_num = bit_huf_cod[i+1:i+3]
                    if temp_num == "00":
                        decod_arr.append(-1)
                    else:
                        decod_arr.append(1)
                    temp_cod = ''
                    i += 3
                elif klass == 1:
                    temp_num = bit_huf_cod[i+1:i+3]
                    if temp_num == "01":
                        decod_arr.append(-2)
                    else:
                        decod_arr.append(2)
                    temp_cod = ''
                    i += 3
                else:
                    temp_num = bit_huf_cod[i+1:i+klass+1]
                    znak = 1
                    if temp_num[0] == "0":
                        temp_num = ''.join('1' if bit == '0' else '0' for bit in temp_num)
                        znak = -1
                    decod_arr.append(int(temp_num, 2) * znak)
                    i += int(reversed_dict[temp_cod]) + 1
                    temp_cod = ''
            else:
                i += 1
        print(f"Decoded DC len={len(decod_arr)}")
        return np.array(decod_arr)

    def cod_Huff_AC(self,arr_ACC):
        class Huffman_root:
            def __init__(self, sim: str, freq: int):
                self.sim = sim
                self.freq = freq
                self.leftchild = None
                self.rightchild = None
            def __lt__(self, other):
                return self.freq < other.freq
            def __str__(self):
                return self.sim + ' ' + str(self.freq)

        def assem_huf_table(tables_huf_cod: dict, root: Huffman_root, link: str):
            if root.sim is None:
                assem_huf_table(tables_huf_cod, root.leftchild, link=link + '0')
                assem_huf_table(tables_huf_cod, root.rightchild, link=link + '1')
            else:
                tables_huf_cod[root.sim] = link
            return tables_huf_cod
        def RLE_zero(s_orig):
            t = 0
            RLE_arr = []
            for i in range(len(s_orig) - 1):
                t += 1
                if s_orig[i] != s_orig[i + 1]:
                    if s_orig[i] == 0:
                        RLE_arr.append([t, int(s_orig[i])])
                        t = 0
                    else:
                        RLE_arr.append([0, int(s_orig[i])])
                        t = 0
                elif s_orig[i] != 0:
                    RLE_arr.append([0, int(s_orig[i])])
                    t = 0
            if s_orig[-1]==0:
                RLE_arr.append([t+1,int(s_orig[-1])])
            else:
                RLE_arr.append([0,int(s_orig[-1])])

            return RLE_arr

        arr_AC = RLE_zero(arr_ACC)
        arr_len = []
        for i in range(len(arr_AC)):
            if arr_AC[i][0] == 0:
                if arr_AC[i][1] == 0:
                    arr_len.append(0)
                else:
                    num = len(bin(abs(arr_AC[i][1]))[2:])
                    arr_len.append(num)
            else:
                num = len(bin(abs(arr_AC[i][0]))[2:])
                arr_len.append(num * -1)

        tables_freq = {}
        for i in arr_len:
            if i in tables_freq:
                tables_freq[i] = tables_freq[i] + 1
            else:
                tables_freq[i] = 1

        queue = [Huffman_root(str(key), tables_freq[key]) for key in tables_freq.keys()]
        heapq.heapify(queue)
        while len(queue) > 1:
            temp_left = heapq.heappop(queue)
            temp_right = heapq.heappop(queue)
            mid = Huffman_root(None, temp_left.freq + temp_right.freq)
            mid.leftchild = temp_left
            mid.rightchild = temp_right
            heapq.heappush(queue, mid)

        tables_huf_cod = {}
        huf_table = assem_huf_table(tables_huf_cod, queue[0], "")
        print(len(huf_table))
        if len(huf_table)==1:
            huf_table={str(arr_len[0]): "1"}
        bit_huf_cod = ""
        for i in range(len(arr_AC)):
            if arr_AC[i][0] == 0:
                if arr_AC[i][1] == 0:
                    num = 0
                else:
                    num = len(bin(abs(arr_AC[i][1]))[2:])
            else:
                num = len(bin(abs(arr_AC[i][0]))[2:]) * -1
            bit_huf_cod += huf_table[str(num)]
            if num == 0:
                bit_huf_cod += '0'
            elif num > 0:
                if arr_AC[i][1] > 0:
                    temp_num = bin(abs(arr_AC[i][1]))[2:]
                else:
                    temp_num = ''.join('1' if bit == '0' else '0' for bit in bin(abs(arr_AC[i][1]))[2:])
                bit_huf_cod += temp_num
            else:
                temp_num = bin(abs(arr_AC[i][0]))[2:]
                bit_huf_cod += temp_num
        print(f"AC Huffman code length={len(bit_huf_cod)}")
        return bit_huf_cod, huf_table
    def decod_Huff_AC(self, bit_huf_cod, huf_table):

        i = 0
        temp_cod = ""
        decod_arr = []
        reversed_dict = {v: k for k, v in huf_table.items()}
        while i < len(bit_huf_cod) - 1:
            temp_cod += bit_huf_cod[i]
            if reversed_dict.get(temp_cod) is not None:
                klass = int(reversed_dict[temp_cod])
                if klass == 0:
                    decod_arr.append(0)
                    i += 2
                    temp_cod = ''
                elif klass > 0:
                    temp_num = bit_huf_cod[i+1:i+klass+1]
                    znak = 1
                    if temp_num[0] == '0':
                        temp_num = ''.join('1' if bit == '0' else '0' for bit in temp_num)
                        znak = -1
                    decod_arr.append(int(temp_num, 2) * znak)
                    i += int(reversed_dict[temp_cod]) + 1
                    temp_cod = ''
                else:
                    temp_num = bit_huf_cod[i+1:i+abs(klass)+1]
                    kol = int(temp_num, 2)
                    for _ in range(kol):
                        decod_arr.append(0)
                    i += int(abs(int(reversed_dict[temp_cod]))) + 1
                    temp_cod = ''
            else:
                i += 1
        print(f"Decoded AC len={len(decod_arr)}")
        return np.array(decod_arr)
    def assembly_DC_and_AC(self, arr_AC, arr_DC, n=8):
        num_blocks = len(arr_DC)
        blocks_per_side = int(math.sqrt(num_blocks))
        matrix = np.zeros((blocks_per_side, blocks_per_side, n, n), dtype=float)
        split_DC = np.zeros_like(arr_DC)
        split_DC[0] = arr_DC[0]
        for i in range(1, len(arr_DC)):
            split_DC[i] = split_DC[i-1] + arr_DC[i]
        split_AC = arr_AC.reshape(num_blocks, (n*n)-1)
        t = 0
        for i in range(blocks_per_side):
            for j in range(blocks_per_side):
                block = np.zeros((n, n))
                block[0, 0] = split_DC[t]
                ac_idx = 0
                row, col = 0, 0
                going_up = True
                while ac_idx < (n*n)-1:
                    if row != 0 or col != 0:
                        block[row, col] = split_AC[t, ac_idx]
                        ac_idx += 1
                    if going_up:
                        if row == 0 and col < n-1:
                            col += 1
                            going_up = False
                        elif col == n-1:
                            row += 1
                            going_up = False
                        else:
                            row -= 1
                            col += 1
                    else:
                        if col == 0 and row < n-1:
                            row += 1
                            going_up = True
                        elif row == n-1:
                            col += 1
                            going_up = True
                        else:
                            row += 1
                            col -= 1
                matrix[i, j] = block
                t += 1
        print(f"Assembled matrix shape={matrix.shape}")
        return matrix

    def ycbcr_to_rgb_image(self, Y, Cb, Cr, output_path):
        if not (Y.shape == Cb.shape == Cr.shape):
            raise ValueError("errir in ycbcr_to_rgb_image")
        height, width = Y.shape
        Y = np.clip(Y, 0, 255)
        Cb = np.clip(Cb, 0, 255)
        Cr = np.clip(Cr, 0, 255)
        ycbcr = np.stack((Y, Cb, Cr), axis=-1)
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        rgb[..., 0] = Y + 1.402 * (Cr - 128)  # Красный канал (R)
        rgb[..., 1] = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)  # Зеленый канал (G)
        rgb[..., 2] = Y + 1.772 * (Cb - 128)  # Синий канал (B)

        print(rgb[0:12,0:12])
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, mode='RGB')
        image.save(output_path)
        print(f"Saved image to {output_path}")
    def upsample_nearest(self, chroma: np.ndarray, scale_y: int = 2, scale_x: int = 2) -> np.ndarray:
        upsampled = chroma.repeat(scale_y, axis=0).repeat(scale_x, axis=1)
        print(f"Upsampled shape={upsampled.shape}")
        return upsampled

    def assembly_block(self, blocks):
     
        reconstructed = blocks.transpose(0, 2, 1, 3).reshape(len(blocks)*8, len(blocks)*8)
        print(f"Assembled blocks shape={reconstructed.shape}")
        return reconstructed







    def write_compressed_data(self, output_path, compressed_data):
        with open(output_path, 'wb') as f:
            # Записываем размеры матриц
            Y_shape = compressed_data['shape']['Y']
            Cb_shape = compressed_data['shape']['Cb']
            Cr_shape = compressed_data['shape']['Cr']
            f.write(struct.pack('!II', Y_shape[0], Y_shape[1]))
            f.write(struct.pack('!II', Cb_shape[0], Cb_shape[1]))
            f.write(struct.pack('!II', Cr_shape[0], Cr_shape[1]))

            # Записываем длины Huffman-кодов
            codes = [
                ('Y', 'dc_code'), ('Y', 'ac_code'),
                ('Cb', 'dc_code'), ('Cb', 'ac_code'),
                ('Cr', 'dc_code'), ('Cr', 'ac_code')
            ]
            for component, code_type in codes:
                code = compressed_data[component][code_type]
                code_len = len(code)
                f.write(struct.pack('!I', code_len))

            # Записываем таблицы Huffman
            tables = [
                ('Y', 'dc_table'), ('Y', 'ac_table'),
                ('Cb', 'dc_table'), ('Cb', 'ac_table'),
                ('Cr', 'dc_table'), ('Cr', 'ac_table')
            ]
            for component, table_type in tables:
                table = compressed_data[component][table_type]
                num_pairs = len(table)
                f.write(struct.pack('!H', num_pairs))
                for symbol, code in table.items():
                    symbol_str = str(symbol)
                    f.write(struct.pack('!B', len(symbol_str)))
                    f.write(symbol_str.encode('ascii'))
                    f.write(struct.pack('!B', len(code)))
                    f.write(code.encode('ascii'))

            # Записываем Huffman-коды
            for component, code_type in codes:
                code = compressed_data[component][code_type]
                pad_bits = (8 - (len(code) % 8)) % 8
                padded_code = code + '0' * pad_bits
                byte_array = bytearray()
                for i in range(0, len(padded_code), 8):
                    byte_str = padded_code[i:i+8]
                    byte_array.append(int(byte_str, 2))
                f.write(byte_array)
        print(f"Compressed data saved to {output_path}")

    def read_compressed_data(self, input_path):
        with open(input_path, 'rb') as f:
            # Читаем размеры матриц
            Y_height, Y_width = struct.unpack('!II', f.read(8))
            Cb_height, Cb_width = struct.unpack('!II', f.read(8))
            Cr_height, Cr_width = struct.unpack('!II', f.read(8))
            shapes = {
                'Y': (Y_height, Y_width),
                'Cb': (Cb_height, Cb_width),
                'Cr': (Cr_height, Cr_width)
            }

            # Читаем длины Huffman-кодов
            codes = [
                ('Y', 'dc_code'), ('Y', 'ac_code'),
                ('Cb', 'dc_code'), ('Cb', 'ac_code'),
                ('Cr', 'dc_code'), ('Cr', 'ac_code')
            ]
            code_lengths = {}
            for component, code_type in codes:
                code_len = struct.unpack('!I', f.read(4))[0]
                code_lengths[f"{component}_{code_type}"] = code_len

            # Читаем таблицы Huffman
            compressed_data = {'shape': shapes}
            tables = [
                ('Y', 'dc_table'), ('Y', 'ac_table'),
                ('Cb', 'dc_table'), ('Cb', 'ac_table'),
                ('Cr', 'dc_table'), ('Cr', 'ac_table')
            ]
            for component, table_type in tables:
                num_pairs = struct.unpack('!H', f.read(2))[0]
                table = {}
                for _ in range(num_pairs):
                    sym_len = struct.unpack('!B', f.read(1))[0]
                    symbol = f.read(sym_len).decode('ascii')
                    code_len = struct.unpack('!B', f.read(1))[0]
                    code = f.read(code_len).decode('ascii')
                    table[symbol] = code
                if component not in compressed_data:
                    compressed_data[component] = {}
                compressed_data[component][table_type] = table

            # Читаем Huffman-коды
            for component, code_type in codes:
                code_len = code_lengths[f"{component}_{code_type}"]
                num_bytes = (code_len + 7) // 8
                byte_array = f.read(num_bytes)
                bit_string = ''
                for byte in byte_array:
                    bit_string += format(byte, '08b')
                bit_string = bit_string[:code_len]
                compressed_data[component][code_type] = bit_string

        print(f"Compressed data loaded from {input_path}")
        return compressed_data

    def compress(self, input_path, quality=50, output_bin_path="compressed_data.bin"):
        # Загрузка изображения
        image_rgb = self.load_image(input_path)
       
        # Преобразование в YCbCr
        image_ycbcr = self.rgb_to_ycbcr(image_rgb)

        print(image_ycbcr[0:12,0:12])
        # Разделение на Y, Cb, Cr
        Y, Cb, Cr = self.sep_matrix_ycbcr(image_ycbcr)
        # Субдискретизация Cb и Cr (4:2:0)
        Cb_down = self.downsample(2, Cb)
        Cr_down = self.downsample(2, Cr)
        # Разделение на блоки 8x8
        Y_blocks = self.split_block(8, Y)
        Cb_blocks = self.split_block(8, Cb_down)
        Cr_blocks = self.split_block(8, Cr_down)
        # Применение DCT
        start = time.time()
        Y_dct = self.matrix_dct(Y_blocks)
        # Y_dct = dctn(Y_blocks)
        end = time.time()
        print("Y_dct time:",end-start)
        start = time.time()
        Cb_dct = self.matrix_dct(Cb_blocks)
        # Cb_dct = dctn(Cb_blocks)
        end = time.time()
        print("Cb_dct time:",end-start)
        start = time.time()
        Cr_dct = self.matrix_dct(Cr_blocks) 
        # Cr_dct = dctn(Cr_blocks) 
        end = time.time()
        print("Cr_dct time:",end-start)
        # Квантование
        start = time.time()
        Y_quant = self.Quant(Y_dct, "Y", quality)
        Cb_quant = self.Quant(Cb_dct, "Cb", quality)
        Cr_quant = self.Quant(Cr_dct, "Cr", quality)
        end = time.time()
        print("All Quant time:",end-start)
        # Разделение на DC и AC
        start = time.time()
        print(Y_quant[0][0][0:10])
        Y_dc, Y_ac = self.split_DC_and_AC(Y_quant)
        Cb_dc, Cb_ac = self.split_DC_and_AC(Cb_quant)
        Cr_dc, Cr_ac = self.split_DC_and_AC(Cr_quant)
        end = time.time()
        print("All split_DC_and_AC time:",end-start)





        # Применяем Huffman-кодирование
        Y_dc_code, Y_dc_table = self.cod_Huff_DC(Y_dc)
        Y_ac_code, Y_ac_table = self.cod_Huff_AC(Y_ac)
        Cb_dc_code, Cb_dc_table = self.cod_Huff_DC(Cb_dc)
        Cb_ac_code, Cb_ac_table = self.cod_Huff_AC(Cb_ac)
        Cr_dc_code, Cr_dc_table = self.cod_Huff_DC(Cr_dc)
        Cr_ac_code, Cr_ac_table = self.cod_Huff_AC(Cr_ac)


        print(f"Y_ac_code={Y_ac_code} Y_ac_code={Y_ac_table}")
        print(f"Cb_ac_code={Cb_ac_code} Cb_ac_table={Cb_ac_table}")
        print(f"Cr_ac_code={Cr_ac_code} Cr_ac_table={Cr_ac_table}")
        # Собираем данные для записи
        compressed_data = {
            "Y": {"dc_code": Y_dc_code, "dc_table": Y_dc_table, "ac_code": Y_ac_code, "ac_table": Y_ac_table},
            "Cb": {"dc_code": Cb_dc_code, "dc_table": Cb_dc_table, "ac_code": Cb_ac_code, "ac_table": Cb_ac_table},
            "Cr": {"dc_code": Cr_dc_code, "dc_table": Cr_dc_table, "ac_code": Cr_ac_code, "ac_table": Cr_ac_table},
            "shape": {"Y": Y.shape, "Cb": Cb.shape, "Cr": Cr.shape}
        }
        # Записываем в файл

        self.write_compressed_data(output_bin_path, compressed_data)
        end = time.time()
        print(f"Compress time: {end - start}")
        return output_bin_path

    def decompress(self, input_bin_path, output_path, quality=50):
        start = time.time()
        # Читаем сжатые данные
        compressed_data = self.read_compressed_data(input_bin_path)

        # Декодируем Huffman-коды
        Y_dc = self.decod_Huff_DC(compressed_data["Y"]["dc_code"], compressed_data["Y"]["dc_table"])
        Y_ac = self.decod_Huff_AC(compressed_data["Y"]["ac_code"], compressed_data["Y"]["ac_table"])
        Cb_dc = self.decod_Huff_DC(compressed_data["Cb"]["dc_code"], compressed_data["Cb"]["dc_table"])
        Cb_ac = self.decod_Huff_AC(compressed_data["Cb"]["ac_code"], compressed_data["Cb"]["ac_table"])
        Cr_dc = self.decod_Huff_DC(compressed_data["Cr"]["dc_code"], compressed_data["Cr"]["dc_table"])
        Cr_ac = self.decod_Huff_AC(compressed_data["Cr"]["ac_code"], compressed_data["Cr"]["ac_table"])
        # Сборка блоков


        Y_blocks = self.assembly_DC_and_AC(Y_ac, Y_dc)
        Cb_blocks = self.assembly_DC_and_AC(Cb_ac, Cb_dc)
        Cr_blocks = self.assembly_DC_and_AC(Cr_ac, Cr_dc)
        # Обратное квантование
        Y_iquant = self.iQuant(Y_blocks, "Y", quality)
        Cb_iquant = self.iQuant(Cb_blocks, "Cb", quality)
        Cr_iquant = self.iQuant(Cr_blocks, "Cr", quality)
        # Применение IDCT
        Y_idct = self.matrix_idct(Y_iquant)
        Cb_idct = self.matrix_idct(Cb_iquant)
        Cr_idct = self.matrix_idct(Cr_iquant)
        # Сборка блоков в матрицы
        Y = self.assembly_block(Y_idct)
        Cb = self.assembly_block(Cb_idct)
        Cr = self.assembly_block(Cr_idct)
        # Апсемплинг Cb и Cr
        Cb_up = self.upsample_nearest(Cb, 2, 2)
        Cr_up = self.upsample_nearest(Cr, 2, 2)
        # Обрезка до исходного размера
        Y = Y[:compressed_data["shape"]["Y"][0], :compressed_data["shape"]["Y"][1]]
        Cb_up = Cb_up[:compressed_data["shape"]["Cb"][0], :compressed_data["shape"]["Cb"][1]]
        Cr_up = Cr_up[:compressed_data["shape"]["Cr"][0], :compressed_data["shape"]["Cr"][1]]
        # Преобразование в RGB и сохранение
        self.ycbcr_to_rgb_image(Y, Cb_up, Cr_up, output_path)
        end = time.time()
        print(f"Decompress time: {end - start}")


if __name__ == "__main__":
    all_start=time.time()
    name_arr=["preview_bw.png","preview_dithered.png","preview_grayscale.png"]
    koef_arr=[0,20,40,60,80,100]
    for name in name_arr:
        for koef in koef_arr:
            start=time.time()
            compressor = JPEGCompressor()
            input_path = str(name)
            output_bin_path = "compressed_"+input_path[0:-4]+"_"+str(koef)+".bin"
            quality = int(koef)
            compressed_data = compressor.compress(input_path, quality=quality,output_bin_path=output_bin_path)
            end=time.time()
            print(f"{input_path} {koef} is completed for: {end-start}")
    all_end=time.time()
    print("all time:",all_end-all_start)
