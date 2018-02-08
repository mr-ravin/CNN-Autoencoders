import cv2 as cv, tensorflow as tf, numpy as np, glob

def preprocess(pre_data_dir='./input/', pos_data_dir='./pros/', dim_width=256, dim_height=256):
    name_list = glob.glob(pre_data_dir + '*.jpg')
    fin_list = []
    for names in name_list:
        tmp_input = cv.imread(str(names))
        pros_name = names.split('/')[-1]
        tmp_grab = cv.resize(tmp_input, (dim_width, dim_height))
        fin_list.append(tmp_grab)
        cv.imwrite(pos_data_dir + pros_name, tmp_grab)


def generate_more_data(diff=10, data_dir='./pros/'):
    name_list = glob.glob(data_dir + '*.jpg')
    for names in name_list:
        if names[-5] != 'r':
            tmp_input_1 = cv.imread(str(names))
            pros_name = names.split('/')[-1]
            pros_addr = names.split('/')[:-1]
            pros_addr = ('/').join(pros_addr) + '/'
            tmp_input_2 = cv.imread(pros_addr + pros_name.split('.')[0] + 'r.jpg')
            rows = tmp_input_1.shape[0]
            cols = tmp_input_1.shape[1]
            chk = 0
            count = -1
            while chk < 360:
                    count = count + 1
                    chk = chk + diff
                    tmp_M = cv.getRotationMatrix2D((cols / 2, rows / 2), chk, 1)
                    dst_1 = cv.warpAffine(tmp_input_1, tmp_M, (cols, rows))
                    cv.imwrite(pros_addr + pros_name.split('.')[0] + str(count) + '.jpg', dst_1)
                    dst_2 = cv.warpAffine(tmp_input_2, tmp_M, (cols, rows))
                    cv.imwrite(pros_addr + pros_name.split('.')[0] + str(count) + 'r.jpg', dst_2)


def generate_numbers(data_dir='./pros/',chk=0):
    name_list = glob.glob(data_dir + '*.jpg')
    input_data = []
    fin_data = []
    if chk ==0:
     for names in name_list:
        if names[-5] != 'r' and names[-5] != 't':
            input_data.append(cv.imread(names))
            fin_data.append(cv.imread(names[:-4] + 'r.jpg'))
    else:
     for names in name_list:
        if names[-5] != 'r' and names[-5] != 't':
            input_data.append(cv.imread(names))
    return (input_data, fin_data)

def CNN_Autoencoders(data_volume, data_volume_fin, epochs=800, batch_size=5, dim_width=256, dim_height=256, nchannels=3):
      print('cnn-ae')
      graph = tf.Graph()
      with graph.as_default():
        sess=tf.Session()
        data = tf.placeholder(tf.float32, [None, dim_width, dim_height, nchannels], name='input_data')
        targets_data = tf.placeholder(tf.float32, [None, dim_width, dim_height, nchannels], name='target_data')
        enc_conv0 = tf.layers.conv2d(data, 64, 3, activation=tf.nn.relu)
        enc_conv0 = tf.layers.max_pooling2d(enc_conv0, 2, 2)
        enc_conv1 = tf.layers.conv2d(enc_conv0, 32, 3, activation=tf.nn.relu)
        enc_conv1 = tf.layers.max_pooling2d(enc_conv1, 2, 2)
        enc_conv2 = tf.layers.conv2d(enc_conv1, 16, 3, activation=tf.nn.relu)
        enc_conv2 = tf.layers.max_pooling2d(enc_conv2, 2, 2)
        ###
        #enc_conv3 = tf.layers.conv2d(enc_conv2,1,2,activation=tf.nn.relu)
        
        """  
        n_nodes_hl1=10
        n_nodes_hl2=5
        n_nodes_hl3=10
        n_classes=int((dim_width/4)*(dim_height/4))
        input_dims=int((dim_width/4)*(dim_height/4))
        hidden_1_layer={'weights':tf.Variable(tf.random_normal([input_dims,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
        hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1 , n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}  
        hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2 , n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
        output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
  
        ########## nw
        l1=tf.add(tf.matmul(inputs_nn,hidden_1_layer['weights']),hidden_1_layer['biases'])
        l1=tf.nn.relu(l1)

        l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
        l2=tf.nn.relu(l2)

        l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
        l3=tf.nn.relu(l3)

        output=tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])
        ########## nw        
        entry_dec=tf.reshape(output,[dim_width/4,dim_height/4])
        """
        #enc_conv4 = tf.layers.conv2d(enc_conv3,1,2,activation=tf.nn.relu)
        ###
        dec_conv2 = tf.image.resize_nearest_neighbor(enc_conv2, tf.constant([64, 64])) #q/4
        dec_conv2 = tf.layers.conv2d(dec_conv2, 32, 3, activation=tf.nn.relu)
        dec_conv1 = tf.image.resize_nearest_neighbor(dec_conv2, tf.constant([128, 128])) #q/2
        dec_conv1 = tf.layers.conv2d(dec_conv1, 64, 3, activation=tf.nn.relu)
        dec_conv0 = tf.image.resize_nearest_neighbor(dec_conv1, tf.constant([256, 256])) #q
        logits =    tf.layers.conv2d(dec_conv0, 3, (3, 3), padding='same', activation=None,name="logits")
        #decoded =   tf.scalar_mul(tf.constant(255.0,tf.float32),decoded_1)
        #decoded = logits
        loss=tf.pow(targets_data - logits, 2)
        #####loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_data, logits=logits)
        cost = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(0.001).minimize(cost)
        sess.run(tf.global_variables_initializer())
        for e_vals in range(epochs):
        
            batch = []
            img=[]
            for bat in range(int(len(data_volume) / batch_size)):
                batch_data = data_volume[bat * batch_size:bat * batch_size + batch_size]
                batch_fin = data_volume_fin[bat * batch_size:bat * batch_size + batch_size]
                print('enter batch')
                img,batch_cost, _ = sess.run([logits,cost, opt ], feed_dict={data: batch_data,targets_data: batch_fin})
                print(('Epoch: {}/{}...').format(e_vals + 1, epochs), ('Training loss: {:.4f}').format(batch_cost))
            ev=img[0]
            ev=ev.astype(int)
            ev[np.where(ev<0)]=0
            cv.imwrite('./restmp/res-'+str(e_vals) + '.jpg', ev)
        #### Inference Code
        infer_name_list = glob.glob("./test/" + '*.jpg')
        infer_data,_=generate_numbers("./test/",1)
        for i in range(len(infer_data)):
          
          tmp_file_name=infer_name_list[i].split("/")[-1]
          tmp2_file_name=tmp_file_name.split(".")[0]
          with tf.variable_scope("logits",reuse=True):
            img = sess.run([logits], feed_dict={data: np.array([infer_data[i]])})
            ev=img[0][0]
            ev=ev.astype(int)
            ev[np.where(ev<0)]=0
            cv.imwrite('./testres/'+str(tmp2_file_name)+'r.jpg', ev)
        #return img
        #### Inference Code


def run(epoch=2):
    data_volume, data_volume_fin = generate_numbers()
    print(np.array(data_volume_fin).shape)
    print(np.array(data_volume).shape)
    CNN_Autoencoders(np.array(data_volume), np.array(data_volume_fin),epoch)
    #rest=CNN_Autoencoders(np.array(data_volume), np.array(data_volume_fin),epoch)
    #return rest
