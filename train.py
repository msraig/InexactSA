import argparse, time
from basic import *

parser = argparse.ArgumentParser()
parser.add_argument('--labeled_step', dest='labeled_step', type=int, default=50000, help='number of iterations with labeled data in the initialization stage')
parser.add_argument('--augment_step', dest='augment_step', type=int, default=50000, help='number of steps in the refinement stage, here a step means an iteration with labeled data and an iteration with unlabeled data')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
parser.add_argument('--loadModel', dest='loadModel', default=None, help='if you want to load a model and continue the training, input the path to the model here')
parser.add_argument('--startStep', dest='startStep',  type=int, default=1, help='the number of the first step, it only affects output')
parser.add_argument('--gpuid', dest='gpuid', default='0,1', help='the value for CUDA_VISIBLE_DEVICES')
parser.add_argument('--testFreq', dest='testFreq', type=int, default=0, help='how many steps to run a test, 0 means never running test')
parser.add_argument('--outputFreq', dest='outputFreq', type=int, default=0, help='the frequency to output test sample images, 20 means output a sample each 20 test samples, 0 means never outputing samples')
parser.add_argument('--saveFreq', dest='saveFreq', type=int, default=10000, help='how many steps to save a model')
parser.add_argument('--ifReal', dest='ifReal', type=int, default=0, help='to train for Perlin data, enter 0; to train for real data, enter 1')

parser.add_argument('--labeled_folder', dest='labeled_folder')
parser.add_argument('--unlabeled_folder', dest='unlabeled_folder', default=None)
parser.add_argument('--lighting_folder', dest='lighting_folder')
parser.add_argument('--test_folder', dest='test_folder', default=None)
parser.add_argument('--output_folder', dest='output_folder')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

setargs(args)

def main(_):
    net = Net_l1()
    start_time = time.time()

    os.makedirs('{}/log'.format(args.output_folder), exist_ok=True)
    os.makedirs('{}/checkpoint'.format(args.output_folder), exist_ok=True)
    os.makedirs('{}/testout'.format(args.output_folder), exist_ok=True)

    writer = tf.summary.FileWriter('{}/log'.format(args.output_folder))

    labeledData = loadData(args.labeled_folder)
    lighting_train = pickle.load(open('{}/trainLighting.dat'.format(args.lighting_folder),'rb'))
    if args.testFreq > 0:
        testData = loadData(args.test_folder)
        if args.ifReal:
            lighting_test = pickle.load(open('{}/realTestLighting.dat'.format(args.lighting_folder),'rb'))
        else:
            lighting_test = pickle.load(open('{}/PerlinTestLighting.dat'.format(args.lighting_folder),'rb'))
    if args.augment_step > 0:
        imgs = loadUnlabeled(args.unlabeled_folder)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(net.init_op)
        saver = tf.train.Saver(max_to_keep=100)
        if not args.loadModel is None:
            print(args.loadModel)
            saver.restore(sess, args.loadModel)
            np.random.seed(int(time.time()))
            

        step = args.startStep
        pt = len(labeledData)
        for i in range(args.labeled_step):
            batchLabeled, pt = getBatchShuffle(labeledData, pt, net, sess, lighting_train)
            _, summ_train, lSp, lRo, lAl, lNm = sess.run([net.train_op,net.summ_op,net.loss_Sp,net.loss_Ro,net.loss_Al,net.loss_Nm], feed_dict={net.inputimage:batchLabeled[0], net.sr_gt:batchLabeled[1], net.albedo_gt:batchLabeled[2], net.normal_gt:batchLabeled[3]})

            if step % 100 == 0:
                print('Labeled step: {}, total: {}s, lossSp: {}, lossRo: {}, lossAl: {}, lossNm: {}'.format(step, time.time()-start_time, lSp, lRo, lAl, lNm))
                writer.add_summary(summ_train,step)

            if(step % args.saveFreq == 0):
                saver.save(sess, '{}/checkpoint/model.ckpt'.format(args.output_folder), step)

            if args.testFreq > 0:
                if(step % args.testFreq == 0):
                    if args.ifReal:
                        summ_test = testModelByBRDF(testData, lighting_test, net, sess, '{}/testout'.format(args.output_folder), args.outputFreq, step, 5)
                    else:
                        summ_test = testModelByBRDF(testData, lighting_test, net, sess, '{}/testout'.format(args.output_folder), args.outputFreq, step, 1)
                    writer.add_summary(summ_test,step)
                
            step = step + 1

        if args.augment_step > 0:
            pt_u = len(imgs)

        for i in range(args.augment_step):
            batchLabeled, pt = getBatchShuffle(labeledData, pt, net, sess, lighting_train)
            _, lSp, lRo, lAl, lNm = sess.run([net.train_op,net.loss_Sp,net.loss_Ro,net.loss_Al,net.loss_Nm], feed_dict={net.inputimage:batchLabeled[0], net.sr_gt:batchLabeled[1], net.albedo_gt:batchLabeled[2], net.normal_gt:batchLabeled[3]})

            if step % 100 == 0:
                print('Labeled step: {}, total: {}s, lossSp: {}, lossRo: {}, lossAl: {}, lossNm: {}'.format(step, time.time()-start_time, lSp, lRo, lAl, lNm))

            image, pt_u = getImageBatch(imgs, pt_u)
            BRDF = sess.run([net.specular_pred,net.roughness_pred,net.albedo_pred,net.normal_pred], feed_dict={net.inputimage:image})
            batchUnlabeled = renderUnlabeled(BRDF, net, sess, lighting_train)
            _, summ_train, lSp, lRo, lAl, lNm = sess.run([net.train_op,net.summ_op,net.loss_Sp,net.loss_Ro,net.loss_Al,net.loss_Nm], feed_dict={net.inputimage:batchUnlabeled[0], net.sr_gt:batchUnlabeled[1], net.albedo_gt:batchUnlabeled[2], net.normal_gt:batchUnlabeled[3]})
        
            if step % 100 == 0:
                print('Unlabeled step: {}, total: {}s, lossSp: {}, lossRo: {}, lossAl: {}, lossNm: {}'.format(step, time.time()-start_time, lSp, lRo, lAl, lNm))
                writer.add_summary(summ_train,step)

            if(step % args.saveFreq == 0):
                saver.save(sess, '{}/checkpoint/model.ckpt'.format(args.output_folder), step)

            if args.testFreq > 0:
                if(step % args.testFreq == 0):
                    if args.ifReal:
                        summ_test = testModelByBRDF(testData, lighting_test, net, sess, '{}/testout'.format(args.output_folder), args.outputFreq, step, 5)
                    else:
                        summ_test = testModelByBRDF(testData, lighting_test, net, sess, '{}/testout'.format(args.output_folder), args.outputFreq, step, 1)
                    writer.add_summary(summ_test,step)
                
            step = step + 1


if __name__ == '__main__':
    tf.app.run()
