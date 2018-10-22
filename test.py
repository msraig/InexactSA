import argparse, time
from basic import *

parser = argparse.ArgumentParser()
parser.add_argument('--loadModel', dest='loadModel', help='the path to the model to test')
parser.add_argument('--gpuid', dest='gpuid', default='0', help='the value for CUDA_VISIBLE_DEVICES')
parser.add_argument('--input_folder', dest='input_folder', help='the input folder containing jpg or png images')
parser.add_argument('--output_folder', dest='output_folder')

args = parser.parse_args()
args.batch_size = 1

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

setargs(args)

def main(_):
    os.makedirs(args.output_folder, exist_ok=True)

    net = Net_l1()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(net.init_op)
        saver = tf.train.Saver()
        saver.restore(sess, args.loadModel)

        inputfiles = glob.glob('{}/*.jpg'.format(args.input_folder)) + glob.glob('{}/*.png'.format(args.input_folder))

        with open('{}/sr.txt'.format(args.output_folder), 'w') as f:

            for inputfile in inputfiles:
                filename = inputfile.split('/')[-1]
                image = toHDR(cv2.resize(cv2.imread(inputfile),(256,256)))[np.newaxis,...]
                alp, nmp, spp, rop = sess.run([net.albedo_pred_test,net.normal_pred_test,net.specular_pred_test,net.roughness_pred_test], feed_dict={net.inputimage:image})

                #cv2.imwrite('{}/{}_albedo.png'.format(args.output_folder, inputfile), toLDR(alp[0]))
                #cv2.imwrite('{}/{}_normal.png'.format(args.output_folder, inputfile), toLDR(nmp[0]))
                save_pfm('{}/{}_albedo.pfm'.format(args.output_folder, filename), alp[0])
                save_pfm('{}/{}_normal.pfm'.format(args.output_folder, filename), nmp[0])
                f.write('{}: {}, {}\n'.format(filename, np.exp(spp[0][0]), np.exp(rop[0][0])))


if __name__ == '__main__':
    tf.app.run()
