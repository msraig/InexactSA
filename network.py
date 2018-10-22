import tensorflow as tf

def HomoNet(image, training, reuse, scope):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        y1 = tf.layers.conv2d(image, 16, 3, strides=(1,1), padding='same', name='conv1', reuse=reuse)
        y1 = tf.layers.batch_normalization(y1, training=training, name='bn1', reuse=reuse)
        y1 = tf.nn.relu(y1)
        y2 = tf.layers.conv2d(y1, 32, 3, strides=(2,2), padding='same', name='conv2', reuse=reuse)
        y2 = tf.layers.batch_normalization(y2, training=training, name='bn2', reuse=reuse)
        y2 = tf.nn.relu(y2)
        y3 = tf.layers.conv2d(y2, 64, 3, strides=(2,2), padding='same', name='conv3', reuse=reuse)
        y3 = tf.layers.batch_normalization(y3, training=training, name='bn3', reuse=reuse)
        y3 = tf.nn.relu(y3)
        y4 = tf.layers.conv2d(y3, 128, 3, strides=(2,2), padding='same', name='conv4', reuse=reuse)
        y4 = tf.layers.batch_normalization(y4, training=training, name='bn4', reuse=reuse)
        y4 = tf.nn.relu(y4)
        y5 = tf.layers.conv2d(y4, 256, 3, strides=(2,2), padding='same', name='conv5', reuse=reuse)
        y5 = tf.layers.batch_normalization(y5, training=training, name='bn5', reuse=reuse)
        y5 = tf.nn.relu(y5)
        y6 = tf.layers.conv2d(y5, 256, 3, strides=(2,2), padding='same', name='conv6', reuse=reuse)
        y6 = tf.layers.batch_normalization(y6, training=training, name='bn6', reuse=reuse)
        y6 = tf.nn.relu(y6)
        y7 = tf.layers.conv2d(y6, 256, 3, strides=(1,1), padding='same', name='conv7', reuse=reuse)
        y7 = tf.layers.batch_normalization(y7, training=training, name='bn7', reuse=reuse)
        y7 = tf.nn.relu(y7)
        y8 = tf.reshape(y7, [-1, 16384])
        y9 = tf.layers.dense(y8, 512, activation=tf.nn.relu, name='fc1', reuse=reuse)
        y10 = tf.layers.dense(y9, 1, activation=None, name='fc2', reuse=reuse)
        
        return y10

def SVNet(image, training, reuse, scope):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        y1 = tf.layers.conv2d(image, 16, 3, strides=(1,1), padding='same', name='conv1', reuse=reuse)
        y1 = tf.layers.batch_normalization(y1, training=training, name='bn1', reuse=reuse)
        y1 = tf.nn.relu(y1)
        y2 = tf.layers.conv2d(y1, 32, 3, strides=(2,2), padding='same', name='conv2', reuse=reuse)
        y2 = tf.layers.batch_normalization(y2, training=training, name='bn2', reuse=reuse)
        y2 = tf.nn.relu(y2)
        y3 = tf.layers.conv2d(y2, 64, 3, strides=(2,2), padding='same', name='conv3', reuse=reuse)
        y3 = tf.layers.batch_normalization(y3, training=training, name='bn3', reuse=reuse)
        y3 = tf.nn.relu(y3)
        y4 = tf.layers.conv2d(y3, 128, 3, strides=(2,2), padding='same', name='conv4', reuse=reuse)
        y4 = tf.layers.batch_normalization(y4, training=training, name='bn4', reuse=reuse)
        y4 = tf.nn.relu(y4)
        y5 = tf.layers.conv2d(y4, 256, 3, strides=(2,2), padding='same', name='conv5', reuse=reuse)
        y5 = tf.layers.batch_normalization(y5, training=training, name='bn5', reuse=reuse)
        y5 = tf.nn.relu(y5)
        y6 = tf.layers.conv2d(y5, 256, 3, strides=(2,2), padding='same', name='conv6', reuse=reuse)
        y6 = tf.layers.batch_normalization(y6, training=training, name='bn6', reuse=reuse)
        y6 = tf.nn.relu(y6)
        y7 = tf.layers.conv2d(y6, 256, 3, strides=(1,1), padding='same', name='conv7', reuse=reuse)
        y7 = tf.layers.batch_normalization(y7, training=training, name='bn7', reuse=reuse)
        y7 = tf.nn.relu(y7)
        y8 = tf.layers.conv2d(y7, 256, 3, strides=(1,1), padding='same', name='conv8', reuse=reuse)
        y8 = tf.layers.batch_normalization(y8, training=training, name='bn8', reuse=reuse)
        y8 = tf.nn.relu(y8)
        y9 = tf.layers.conv2d(y8, 256, 3, strides=(1,1), padding='same', name='conv9', reuse=reuse)
        y9 = tf.layers.batch_normalization(y9, training=training, name='bn9', reuse=reuse)
        y9 = tf.nn.relu(y9)
        y = tf.concat([y9,y6], axis=3) #8*8*512
        y = tf.layers.conv2d(y, 256, 3, strides=(1,1), padding='same', name='conv10', reuse=reuse) #8*8*256
        y = tf.layers.batch_normalization(y, training=training, name='bn10', reuse=reuse)
        y = tf.nn.relu(y)
        y = tf.image.resize_bilinear(y, [16,16])
        y = tf.concat([y,y5], axis=3) #16*16*512
        y = tf.layers.conv2d(y, 128, 3, strides=(1,1), padding='same', name='conv11', reuse=reuse) #16*16*128
        y = tf.layers.batch_normalization(y, training=training, name='bn11', reuse=reuse)
        y = tf.nn.relu(y)
        y = tf.image.resize_bilinear(y, [32,32]) #32*32*128
        y = tf.concat([y,y4], axis=3) #32*32*256
        y = tf.layers.conv2d(y, 64, 3, strides=(1,1), padding='same', name='conv12', reuse=reuse) #32*32*64
        y = tf.layers.batch_normalization(y, training=training, name='bn12', reuse=reuse)
        y = tf.nn.relu(y)
        y = tf.image.resize_bilinear(y, [64,64]) #64*64*64
        y = tf.concat([y,y3], axis=3)
        y = tf.layers.conv2d(y, 32, 3, strides=(1,1), padding='same', name='conv13', reuse=reuse)
        y = tf.layers.batch_normalization(y, training=training, name='bn13', reuse=reuse)
        y = tf.nn.relu(y)
        y = tf.image.resize_bilinear(y, [128,128])
        y = tf.concat([y,y2], axis=3)
        y = tf.layers.conv2d(y, 16, 3, strides=(1,1), padding='same', name='conv14', reuse=reuse)
        y = tf.layers.batch_normalization(y, training=training, name='bn14', reuse=reuse)
        y = tf.nn.relu(y)
        y = tf.image.resize_bilinear(y, [256,256]) #256*256*16
        y = tf.concat([y,y1], axis=3)
        y = tf.layers.conv2d(y, 16, 3, strides=(1,1), padding='same', name='conv15', reuse=reuse)
        y = tf.layers.batch_normalization(y, training=training, name='bn15', reuse=reuse)
        y = tf.nn.relu(y)
        y = tf.layers.conv2d(y, 3, 3, strides=(1,1), padding='same', activation=tf.nn.sigmoid, name='conv16', reuse=reuse)
        
        return y
