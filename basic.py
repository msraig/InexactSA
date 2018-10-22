from utils import *
from network import *
import os.path
import pickle
import glob
import cv2
from transforms3d.axangles import axangle2aff, axangle2mat
from _render_all_fast_multiGPU import whiteBalanceRenderWithMask as render
from skimage.measure import compare_ssim as ssim

def setargs(args):
    global batch_size
    batch_size = args.batch_size
    np.random.seed(258)


def loadData(folder):
    synData = []
    with open('{}/sr.txt'.format(folder)) as f:
        i = 0
        while True:
            line = f.readline()
            if len(line) < 2:
                break
            s = float(line.split(' ')[1])
            r = float(line.split(' ')[2])
            al = load_pfm('{}/{}_albedo.pfm'.format(folder, i))
            nm = load_pfm('{}/{}_normal.pfm'.format(folder, i))
            synData += [[s,r,al,nm]]
            i += 1
    return synData


def loadUnlabeled(folder):
    unlabeledData = []
    i = 0

    if os.path.isfile('{}/0_render.pfm'.format(folder)):
        while True:
            if os.path.isfile('{}/{}_render.pfm'.format(folder, i)):
                unlabeledData += [load_pfm('{}/{}_render.pfm'.format(folder, i))]
                i += 1
            else:
                break
    elif os.path.isfile('{}/0.jpg'.format(folder)):
        while True:
            if os.path.isfile('{}/{}.jpg'.format(folder, i)):
                unlabeledData += [toHDR(cv2.imread('{}/{}.jpg'.format(folder, i)))]
                i += 1
            else:
                break
    return unlabeledData


def getBatchShuffle(synData, pt, net, sess, lighting=None):
    BRDF = []
    for b in range(batch_size):
        if pt >= len(synData):
            np.random.shuffle(synData)
            nms = []
            sps = []
            ros = []
            for i in range(len(synData)):
                nms += [synData[i][3]]
                sps += [synData[i][0]]
                ros += [synData[i][1]]
            np.random.shuffle(nms)
            np.random.shuffle(sps)
            np.random.shuffle(ros)
            for i in range(len(synData)):
                synData[i][3] = nms[i]
                synData[i][0] = sps[i]
                synData[i][1] = ros[i]
            pt = 0
        BRDF += [synData[pt]]
        pt += 1

    albedo = np.zeros([batch_size,256,256,3])
    specular = np.zeros([batch_size,256,256,3])
    roughness = np.zeros([batch_size,256,256])
    normalxy = np.zeros([batch_size,256,256,2])
    mask = np.ones([batch_size,256,256])
    lid = np.zeros([batch_size],int)
    matl = np.zeros([batch_size,4,4])
    sr = np.zeros([batch_size,2])
    normal = np.zeros([batch_size,256,256,3])
    for b in range(batch_size):
        if lighting is None:
            lid[b] = np.random.randint(net.num_light)
            matl[b] = np.matmul(axangle2aff([1.0,0.0,0.0],(np.random.rand()-0.5)*np.pi/6), axangle2aff([0.0,1.0,0.0],np.random.rand()*np.pi*2))
        else:
            selection = np.random.randint(len(lighting[0]))
            lid[b] = lighting[0][selection]
            matl[b] = lighting[1][selection]
        specular[b] = np.full([256,256,3], BRDF[b][0])
        roughness[b] = np.full([256,256], BRDF[b][1])
        albedo[b] = BRDF[b][2]
        normal[b] = BRDF[b][3]
        normalxy[b] = normal[b][:,:,2:0:-1]*2-1
        sr[b,0] = BRDF[b][0]
        sr[b,1] = BRDF[b][1]
    image_out = sess.run(net.render_batch, {net.albedo_batch:albedo, net.specular_batch:specular, net.roughness_batch:roughness, net.normalxy_batch:normalxy, net.mask_batch:mask, net.lightID_batch:lid, net.matLight_batch:matl})
    batchSyn = (image_out, np.log(sr), albedo, normal)
    return (batchSyn, pt)


def getImageBatch(imgs, pt_u):
    image = np.zeros([batch_size,256,256,3])
    num_img = len(imgs)
    for b in range(batch_size):
        if (pt_u >= num_img):
            np.random.shuffle(imgs)
            pt_u = 0
        image[b] = imgs[pt_u]
        pt_u += 1
    return((image,pt_u))


def renderUnlabeled(BRDF, net, sess, lighting=None):
    sp = np.minimum(np.maximum(np.exp(BRDF[0]),0.001),1.0)
    bs = np.shape(sp)[0]
    if bs>0:
        ro = np.minimum(np.maximum(np.exp(BRDF[1]),0.001),1.0)
        albedo = np.minimum(np.maximum(BRDF[2],0.001),1.0)
        nm = BRDF[3] * 2 - 1
        normal = (nm / np.stack([np.sqrt(nm[...,0]**2+nm[...,1]**2+nm[...,2]**2)]*3, axis=-1) + 1) / 2
        specular = np.zeros([bs,256,256,3])
        roughness = np.zeros([bs,256,256])
        normalxy = np.zeros([bs,256,256,2])
        mask = np.ones([bs,256,256])
        lid = np.zeros([bs],int)
        matl = np.zeros([bs,4,4])
        for b in range(bs):
            if lighting is None:
                lid[b] = np.random.randint(net.num_light)
                matl[b] = np.matmul(axangle2aff([1.0,0.0,0.0],(np.random.rand()-0.5)*np.pi/6), axangle2aff([0.0,1.0,0.0],np.random.rand()*np.pi*2))
            else:
                selection = np.random.randint(len(lighting[0]))
                lid[b] = lighting[0][selection]
                matl[b] = lighting[1][selection]
            specular[b] = np.full([256,256,3], sp[b][0])
            roughness[b] = np.full([256,256], ro[b][0])
            normalxy[b] = normal[b][:,:,2:0:-1]*2-1
        image_out = sess.run(net.render_batch, {net.albedo_batch:albedo, net.specular_batch:specular, net.roughness_batch:roughness, net.normalxy_batch:normalxy, net.mask_batch:mask, net.lightID_batch:lid, net.matLight_batch:matl})
        batchUnlabeled = (image_out, np.log(np.concatenate((sp,ro),axis=1)), albedo, normal)
        return(batchUnlabeled)
    else:
        return(None)


def testModelByBRDF(synData, lighting, net, sess, root, outputFreq=0, step=0, times=1):
    splt = 0
    rolt = 0
    ablt = 0
    nmlt = 0
    relt = 0
    stlt = 0
    nllt = 0
    cnt = 0
    pt = 0
    if not lighting is None:
        lids, matls = lighting
    for i in range(len(synData)*times//batch_size):
        BRDF = []
        for b in range(batch_size):
            if pt >= len(synData)*times:
                pt = 0
            BRDF += [synData[pt//times]]
            pt += 1

        albedo = np.zeros([batch_size,256,256,3])
        specular = np.zeros([batch_size,256,256,3])
        roughness = np.zeros([batch_size,256,256])
        normalxy = np.zeros([batch_size,256,256,2])
        mask = np.ones([batch_size,256,256])
        lid = np.zeros([batch_size],int)
        matl = np.zeros([batch_size,4,4])
        sr = np.zeros([batch_size,2])
        normal = np.zeros([batch_size,256,256,3])
        for b in range(batch_size):
            if not lighting is None:
                lid[b] = lids[i*batch_size+b]
                matl[b] = matls[i*batch_size+b]
            else:
                lid[b] = np.random.randint(net.num_light)
                matl[b] = np.matmul(axangle2aff([1.0,0.0,0.0],(np.random.rand()-0.5)*np.pi/6), axangle2aff([0.0,1.0,0.0],np.random.rand()*np.pi*2))
            specular[b] = np.full([256,256,3], BRDF[b][0])
            roughness[b] = np.full([256,256], BRDF[b][1])
            albedo[b] = BRDF[b][2]
            normal[b] = BRDF[b][3]
            normalxy[b] = normal[b][:,:,2:0:-1]*2-1
            sr[b,0] = BRDF[b][0]
            sr[b,1] = BRDF[b][1]
        image_out = sess.run(net.render_batch, {net.albedo_batch:albedo, net.specular_batch:specular, net.roughness_batch:roughness, net.normalxy_batch:normalxy, net.mask_batch:mask, net.lightID_batch:lid, net.matLight_batch:matl})
        batchSyn = (image_out, np.log(sr), albedo, normal)
        spl, rol, abl, nml, spp, rop, alp, nmp = sess.run([net.loss_Sp_test, net.loss_Ro_test, net.loss_Al_test, net.loss_Nm_test, net.specular_pred_test, net.roughness_pred_test, net.albedo_pred_test, net.normal_pred_test], feed_dict={net.inputimage:batchSyn[0], net.sr_gt:batchSyn[1], net.albedo_gt:batchSyn[2], net.normal_gt:batchSyn[3]})
        splt += spl
        rolt += rol
        ablt += abl
        nmlt += nml

        for b in range(batch_size):
            specular[b] = np.full([256,256,3], np.exp(spp[b]))
            roughness[b] = np.full([256,256], np.exp(rop[b]))
            nm = nmp[b] * 2 - 1
            normalxy[b] = (nm / np.stack([np.sqrt(nm[...,0]**2+nm[...,1]**2+nm[...,2]**2)]*3, axis=-1))[:,:,2:0:-1]
        rerender_out = sess.run(net.render_batch, {net.albedo_batch:alp, net.specular_batch:specular, net.roughness_batch:roughness, net.normalxy_batch:normalxy, net.mask_batch:mask, net.lightID_batch:lid, net.matLight_batch:matl})

        for b in range(batch_size):
            rel = np.mean((rerender_out[b]-image_out[b])**2)
            relt += rel
            dssim = 1 - ssim(toLDR(rerender_out[b]).astype(np.uint8), toLDR(image_out[b]).astype(np.uint8), data_range=255, multichannel=True)
            stlt += dssim
            if outputFreq > 0:
                if (cnt*batch_size+b) % outputFreq == 0:
                    saveRerender('{}/{}_{}.png'.format(root,step,cnt*batch_size+b), rerender_out[b], image_out[b], alp[b], batchSyn[2][b], nmp[b], batchSyn[3][b],np.exp(spp[b][0]),np.exp(batchSyn[1][b][0]),np.exp(rop[b][0]),np.exp(batchSyn[1][b][1]),rel,dssim)
        cnt += 1

    print('Test: avg sp_loss: {}, avg ro_loss: {}, avg al_loss: {}, avg nm_loss: {}, avg rerender_loss: {}, avg dssim: {}'.format(splt/cnt,rolt/cnt,ablt/cnt,nmlt/cnt,relt/cnt/batch_size,stlt/cnt/batch_size))
    summ_test = sess.run(net.summ_op_test, feed_dict={net.phloss_Sp_test:splt/cnt, net.phloss_Ro_test:rolt/cnt, net.phloss_Al_test:ablt/cnt, net.phloss_Nm_test:nmlt/cnt})
    return(summ_test)



class Net_l1:
    def __init__(self):
        def xyzToThetaPhi_tf(input_array):
            normalized_array = input_array / tf.stack([tf.sqrt(input_array[...,0]**2+input_array[...,1]**2+input_array[...,2]**2)]*3, axis=-1)
            theta_array = tf.acos(normalized_array[..., 2])
            phi_array = tf.atan2(normalized_array[..., 1], normalized_array[..., 0])
            return tf.stack((theta_array, phi_array), axis = -1)

        global_step = tf.train.create_global_step()
        learning_rate = tf.train.inverse_time_decay(0.001, global_step, 1, 0.0001)

        self.inputimage = tf.placeholder(tf.float32, [batch_size,256,256,3])
        self.sr_gt = tf.placeholder(tf.float32, [batch_size,2])
        with tf.device('/device:GPU:0'):
            self.specular_pred = HomoNet(self.inputimage, True, False, 'SpNet')
            self.albedo_pred = SVNet(self.inputimage, True, False, 'AlNet')
            self.albedo_gt = tf.placeholder(tf.float32, [batch_size,256,256,3])
            self.loss_Sp = tf.reduce_mean(tf.abs(self.specular_pred[:,0]-self.sr_gt[:,0]))
            self.loss_Al = tf.reduce_mean(tf.abs(self.albedo_pred-self.albedo_gt))
            summ_Sp = tf.summary.scalar('loss_Sp', self.loss_Sp)
            summ_Al = tf.summary.scalar('loss_Al', self.loss_Al)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_0 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_Sp+self.loss_Al, global_step)
        with tf.device('/device:GPU:1'):
            self.roughness_pred = HomoNet(self.inputimage, True, False, 'RoNet')
            self.normal_pred = SVNet(self.inputimage, True, False, 'NmNet')
            self.normal_gt = tf.placeholder(tf.float32, [batch_size,256,256,3])
            self.loss_Ro = tf.reduce_mean(tf.abs(self.roughness_pred[:,0]-self.sr_gt[:,1]))
            self.loss_Nm = tf.reduce_mean(tf.abs(self.normal_pred-self.normal_gt))
            summ_Ro = tf.summary.scalar('loss_Ro', self.loss_Ro)
            summ_Nm = tf.summary.scalar('loss_Nm', self.loss_Nm)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='RoNet') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='NmNet')
            with tf.control_dependencies(update_ops):
                train_op_1 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_Ro+self.loss_Nm, global_step)
        self.train_op = tf.group(train_op_0,train_op_1)
        self.summ_op = tf.summary.merge([summ_Sp,summ_Ro,summ_Al,summ_Nm])


        halfbs = batch_size // 2
        matWorld = tf.constant(np.tile(np.identity(4),[halfbs,1,1]), tf.float32)
        fovRadian = 60.0 / 180.0 * math.pi
        cameraDist = 1.0  / (math.tan(fovRadian / 2.0))
        eyePosWorld = tf.constant(np.array([0, 0, cameraDist]).astype(np.float32), tf.float32)
        matView = tf.constant(lookAt(np.array([0, 0, cameraDist]), np.array([0,0,0.0]), np.array([0,1.0,0])).astype(np.float32), tf.float32)
        matProj = tf.constant(perspective(fovRadian, 1.0, 0.01, 100.0), tf.float32)
    
        self.albedo_batch = tf.placeholder(tf.float32, [batch_size,256,256,3])
        self.specular_batch = tf.placeholder(tf.float32, [batch_size,256,256,3])
        self.roughness_batch = tf.placeholder(tf.float32, [batch_size,256,256])
        self.normalxy_batch = tf.placeholder(tf.float32, [batch_size,256,256,2])
        self.mask_batch = tf.placeholder(tf.float32, [batch_size,256,256])
        self.lightID_batch = tf.placeholder(tf.int32, [batch_size])
        self.matLight_batch = tf.placeholder(tf.float32, [batch_size,4,4])

        light_folder = "/home/D/v-wenye/SAN/SA_SVBRDF_Net_Code/envMaps"
        light_file = "light.txt"
        self.num_light = 49

        with tf.device('/device:GPU:0'):
            render_batch0 = render(self.albedo_batch[0:halfbs],self.specular_batch[0:halfbs],self.roughness_batch[0:halfbs],self.normalxy_batch[0:halfbs],self.mask_batch[0:halfbs],self.lightID_batch[0:halfbs],self.matLight_batch[0:halfbs],
                                    matWorld,eyePosWorld,matView,matProj,light_folder,light_file,
                                    num_sample_spec=1024,use_fresnel=0,instance_id='wbrender0')

        with tf.device('/device:GPU:1'):
            render_batch1 = render(self.albedo_batch[halfbs:batch_size],self.specular_batch[halfbs:batch_size],self.roughness_batch[halfbs:batch_size],self.normalxy_batch[halfbs:batch_size],self.mask_batch[halfbs:batch_size],self.lightID_batch[halfbs:batch_size],self.matLight_batch[halfbs:batch_size],
                                    matWorld,eyePosWorld,matView,matProj,light_folder,light_file,
                                    num_sample_spec=1024,use_fresnel=0,instance_id='wbrender1')
        self.render_batch = tf.concat([render_batch0,render_batch1],0)


        with tf.device('/device:GPU:0'):
            self.specular_pred_test = HomoNet(self.inputimage, False, True, 'SpNet')
            self.albedo_pred_test = SVNet(self.inputimage, False, True, 'AlNet')
            self.loss_Sp_test = tf.reduce_mean(tf.abs(self.specular_pred_test[:,0]-self.sr_gt[:,0]))
            self.loss_Al_test = tf.reduce_mean(tf.abs(self.albedo_pred_test-self.albedo_gt))
        with tf.device('/device:GPU:1'):
            self.roughness_pred_test = HomoNet(self.inputimage, False, True, 'RoNet')
            self.normal_pred_test = SVNet(self.inputimage, False, True, 'NmNet')
            self.loss_Ro_test = tf.reduce_mean(tf.abs(self.roughness_pred_test[:,0]-self.sr_gt[:,1]))
            self.loss_Nm_test = tf.reduce_mean(tf.abs(self.normal_pred_test-self.normal_gt))

        self.phloss_Sp_test = tf.placeholder(tf.float32,[])
        self.phloss_Ro_test = tf.placeholder(tf.float32,[])
        self.phloss_Al_test = tf.placeholder(tf.float32,[])
        self.phloss_Nm_test = tf.placeholder(tf.float32,[])
        summ_Sp_test = tf.summary.scalar('loss_Sp_test', self.phloss_Sp_test)
        summ_Ro_test = tf.summary.scalar('loss_Ro_test', self.phloss_Ro_test)
        summ_Al_test = tf.summary.scalar('loss_Al_test', self.phloss_Al_test)
        summ_Nm_test = tf.summary.scalar('loss_Nm_test', self.phloss_Nm_test)
        self.summ_op_test = tf.summary.merge([summ_Sp_test,summ_Ro_test,summ_Al_test,summ_Nm_test])

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

