import pickle
import numpy as np
from transforms3d.axangles import axangle2aff, axangle2mat
import tensorflow as tf
from _render_all_fast_multiGPU import whiteBalanceRenderWithMask as render
import math
from utils import *
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if len(sys.argv) != 4:
    print('Invalid input! Please input: labeled data folder, output folder, lighting folder')
    sys.exit()

os.makedirs(sys.argv[2], exist_ok=True)

light_folder = sys.argv[3]
light_file = "light.txt"
num_light = 49
lighting = pickle.load(open('{}/trainLighting.dat'.format(light_folder),'rb'))

srData = []
with open('{}/sr.txt'.format(sys.argv[1])) as f:
    while True:
        line = f.readline()
        if len(line) < 2:
            break
        s = float(line.split(' ')[1])
        r = float(line.split(' ')[2])
        srData += [[s,r]]

batch_size = 8
matWorld = tf.constant(np.tile(np.identity(4),[batch_size,1,1]), tf.float32)
fovRadian = 60.0 / 180.0 * math.pi
cameraDist = 1.0  / (math.tan(fovRadian / 2.0))
eyePosWorld = tf.constant(np.array([0, 0, cameraDist]).astype(np.float32), tf.float32)
matView = tf.constant(lookAt(np.array([0, 0, cameraDist]), np.array([0,0,0.0]), np.array([0,1.0,0])).astype(np.float32), tf.float32)
matProj = tf.constant(perspective(fovRadian, 1.0, 0.01, 100.0), tf.float32)

albedo_batch = tf.placeholder(tf.float32, [batch_size,256,256,3])
specular_batch = tf.placeholder(tf.float32, [batch_size,256,256,3])
roughness_batch = tf.placeholder(tf.float32, [batch_size,256,256])
normalxy_batch = tf.placeholder(tf.float32, [batch_size,256,256,2])
mask_batch = tf.placeholder(tf.float32, [batch_size,256,256])
lightID_batch = tf.placeholder(tf.int32, [batch_size])
matLight_batch = tf.placeholder(tf.float32, [batch_size,4,4])


render_batch = render(albedo_batch,specular_batch,roughness_batch,normalxy_batch,mask_batch,lightID_batch,matLight_batch,
                        matWorld,eyePosWorld,matView,matProj,light_folder,light_file,
                        num_sample_spec=1024,use_fresnel=0)

albedo = np.zeros([batch_size,256,256,3])
specular = np.zeros([batch_size,256,256,3])
roughness = np.zeros([batch_size,256,256])
normalxy = np.zeros([batch_size,256,256,2])
mask = np.ones([batch_size,256,256])
lightID = np.zeros([batch_size],int)
matLight = np.zeros([batch_size,4,4])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    for j in range(len(srData)//1000):
        synData = []
        for i in range(1000):
            al = load_pfm('{}/{}_albedo.pfm'.format(sys.argv[1], j*1000+i))
            nm = load_pfm('{}/{}_normal.pfm'.format(sys.argv[1], j*1000+i))
            #if i<50 or i>=100:
            synData += [[srData[j*1000+i][0],srData[j*1000+i][1],al,nm]]

        per = np.random.permutation(1000)
        nms = []
        for i in range(1000):
            nms += [synData[i][3]]

        for i in range(1000):
            synData[i][3] = nms[per[i]]

        for i in range(1000//batch_size):
            for b in range(batch_size):
                selection = np.random.randint(len(lighting[0]))
                lightID[b] = lighting[0][selection]
                matLight[b] = lighting[1][selection]
                albedo[b] = synData[i*batch_size+b][2]
                normalxy[b] = synData[i*batch_size+b][3][:,:,2:0:-1]*2-1
                specular[b] = np.full([256,256,3], synData[i*batch_size+b][0])
                roughness[b] = np.full([256,256], synData[i*batch_size+b][1])

            image_out = sess.run(render_batch, {albedo_batch:albedo, specular_batch:specular, roughness_batch:roughness, normalxy_batch:normalxy, mask_batch:mask, lightID_batch:lightID, matLight_batch:matLight})
            for b in range(batch_size):
                save_pfm('{}/{}_render.pfm'.format(sys.argv[2], j*1000+i*batch_size+b), image_out[b])
