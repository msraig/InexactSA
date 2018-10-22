import math, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def save_pfm(filepath, img, reverse = 1):
    color = None
    file = open(filepath, 'wb')
    if(img.dtype.name != 'float32'):
        img = img.astype(np.float32)

    color = True if (len(img.shape) == 3) else False

    if(reverse and color):
        img = img[:,:,::-1]

    img = img[::-1,...]

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (img.shape[1], img.shape[0]))
    
    endian = img.dtype.byteorder
    scale = 1.0
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)
    img.tofile(file)
    file.close()

def load_pfm(filepath, reverse = 1):
    file = open(filepath, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    color = (header == b'PF')

    width, height = map(int, file.readline().strip().decode('ascii').split(' '))
    scale = float(file.readline().rstrip().decode('ascii'))
    endian = '<' if(scale < 0) else '>'
    scale = abs(scale)

    rawdata = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    file.close()

    if(color):  
        return rawdata.reshape(shape).astype(np.float32)[::-1,:,::-1]
    else:
        return rawdata.reshape(shape).astype(np.float32)[::-1,:]
    

def xyzToThetaPhi(input_array):
    normalized_array = input_array / np.stack([np.sqrt(input_array[...,0]**2+input_array[...,1]**2+input_array[...,2]**2)]*3, axis=-1)
    theta_array = np.arccos(normalized_array[..., 2])
    phi_array = np.arctan2(normalized_array[..., 1], normalized_array[..., 0])
    return np.stack((theta_array, phi_array), axis = -1)

def thetaPhiToXYZ(input_array):
    x_array = np.sin(input_array[..., 0]) * np.cos(input_array[..., 1])
    y_array = np.sin(input_array[..., 0]) * np.sin(input_array[..., 1])
    z_array = np.cos(input_array[..., 0])

    return np.stack((x_array, y_array, z_array), axis = -1)


def lookAt(eyePos, center, up):
    eyePos = np.array(eyePos)
    center = np.array(center)
    up = np.array(up)

    f = center - eyePos
    f = f / np.linalg.norm(f)
    
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    out = np.eye(4)
    out[0, 0:3] = s
    out[1, 0:3] = u
    out[2, 0:3] = -f
    out[0, 3] = -np.dot(s, eyePos)
    out[1, 3] = -np.dot(u, eyePos)
    out[2, 3] = -np.dot(f, eyePos)

    return out


def perspective(fovy, aspect, zNear, zFar):
    tanHalfFovy = math.tan(fovy / 2.0)

    out = np.zeros((4,4))
    out[0, 0] = 1.0 / (aspect * tanHalfFovy)
    out[1, 1] = 1.0 / tanHalfFovy
    out[2, 2] = zFar / (zNear - zFar)
    out[3, 2] = -1.0
    out[2, 3] = -(zFar*zNear) / (zFar - zNear)

    return out


def saveRerender(file, rerender, render, alpd, algt, nmpd, nmgt, sppd, spgt, ropd, rogt, rel, dssim):
    image = np.zeros((582,788,3), np.uint8)
    image[0:256, 0:256] = toLDR(rerender)[:,:,::-1]
    image[266:522, 0:256] = toLDR(render)[:,:,::-1]
    image[0:256, 266:522] = toLDR(alpd)[:,:,::-1]
    image[266:522, 266:522] = toLDR(algt)[:,:,::-1]
    image[0:256, 532:788] = toLDR(nmpd)[:,:,::-1]
    image[266:522, 532:788] = toLDR(nmgt)[:,:,::-1]

    img = Image.fromarray(image, 'RGB')
    draw = ImageDraw.Draw(img)
    text = 'sppd: {:6f}   ropd: {:6f}   rrel: {:6f}\nspgt: {:6f}   rogt: {:6f}   dssi: {:6f}'.format(sppd,ropd,rel,spgt,rogt,dssim)
    font = ImageFont.truetype('/FreeMono.ttf',25)
    draw.multiline_text((10,532), text, 'white', font)
    img.save(file, 'PNG')


def toHDR(img):
    img = img / 255.0
    img_out = img ** (2.2)
    return img_out.astype(np.float32)

def toLDR(img, scale = 1.0):
    img_out = scale * img ** (1.0 / 2.2)
    img_out = np.minimum(255, img_out * 255)
    return img_out.astype(np.uint8)