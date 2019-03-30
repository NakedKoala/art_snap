from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import vgg16
import subprocess
import boto3
import os
import glob



def stylize(content_img, style_img, base_img=None, saveto=None, gif_step=5,
            n_iterations=100, style_weight=2.0, content_weight=1.0):
    """Stylization w/ the given content and style images.

    Follows the approach in Leon Gatys et al.

    Parameters
    ----------
    content_img : np.ndarray
        Image to use for finding the content features.
    style_img : TYPE
        Image to use for finding the style features.
    base_img : None, optional
        Image to use for the base content.  Can be noise or an existing image.
        If None, the content image will be used.
    saveto : str, optional
        Name of GIF image to write to, e.g. "stylization.gif"
    gif_step : int, optional
        Modulo of iterations to save the current stylization.
    n_iterations : int, optional
        Number of iterations to run for.
    style_weight : float, optional
        Weighting on the style features.
    content_weight : float, optional
        Weighting on the content features.

    Returns
    -------
    stylization : np.ndarray
        Final iteration of the stylization.
    """
    # Preprocess both content and style images
    assert(len(content_img.shape) == 4)
    assert(style_weight==2.0)
    content_img = np.array(list(map(vgg_preprocess, content_img)))
    style_img = vgg16.preprocess(style_img, dsize=(224, 224))[np.newaxis]
    if base_img is None:
        base_img = content_img
    else:
        base_img = np.zeros(content_img.shape)
        base_img =  np.array(list(map(vgg_preprocess, base_img)))
        
    assert(base_img.shape == content_img.shape)
    
    # Get Content and Style features
    net = vgg16.get_vgg_model()
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.import_graph_def(net['graph_def'], name='vgg')
        names = [op.name for op in g.get_operations()]
        x = g.get_tensor_by_name(names[0] + ':0')
        content_layer = 'vgg/conv3_2/conv3_2:0'
        content_features = g.get_tensor_by_name(
            content_layer).eval(feed_dict={
                x: content_img,
                'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
                'vgg/dropout/random_uniform:0': [[1.0] * 4096]})
#         print(f'content features shape: {content_features.shape}')
        style_layers = ['vgg/conv1_1/conv1_1:0',
                        'vgg/conv2_1/conv2_1:0',
                        'vgg/conv3_1/conv3_1:0',
                        'vgg/conv4_1/conv4_1:0',
                        'vgg/conv5_1/conv5_1:0']
        style_activations = []
        for style_i in style_layers:
            style_activation_i = g.get_tensor_by_name(style_i).eval(
                feed_dict={
                    x: style_img,
                    'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
                    'vgg/dropout/random_uniform:0': [[1.0] * 4096]})
            style_activations.append(style_activation_i)
        style_features = []
        for style_activation_i in style_activations:
            s_i = np.reshape(style_activation_i,
                             [-1, style_activation_i.shape[-1]])
            gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
            style_features.append(gram_matrix.astype(np.float32))       

    # Define content and style loss
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
       
        net_input = tf.to_float(tf.Variable(base_img))

        tf.import_graph_def(
            net['graph_def'],
            name='vgg',
            input_map={'images:0': net_input})
    
        content_loss = tf.nn.l2_loss( (g.get_tensor_by_name(content_layer) -
                                      content_features) /
                                       content_features.size   )
    
        style_loss = np.float32(0.0)
        for style_layer_i, style_gram_i in zip(style_layers, style_features):
            layer_i = g.get_tensor_by_name(style_layer_i)
            layer_shape = layer_i.get_shape().as_list()
            layer_size =layer_shape[0] * layer_shape[1] * layer_shape[2] * layer_shape[3]
            layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
            gram_matrix = tf.matmul(
                tf.transpose(layer_flat), layer_flat) / layer_size
            style_loss = tf.add(
                style_loss, tf.nn.l2_loss(
                    (gram_matrix - style_gram_i) /
                    np.float32(style_gram_i.size)))
            
        
        loss = content_weight * content_loss + style_weight * style_loss

# Optimize base_img
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        sess.run(tf.global_variables_initializer())
        
        imgs = []
        for it_i in range(n_iterations):
            _, this_loss, synth = sess.run(
                [optimizer, loss, net_input],
                feed_dict={
                    'vgg/dropout_1/random_uniform:0': np.ones(
                        g.get_tensor_by_name(
                            'vgg/dropout_1/random_uniform:0'
                        ).get_shape().as_list()),
                    'vgg/dropout/random_uniform:0': np.ones(
                        g.get_tensor_by_name(
                            'vgg/dropout/random_uniform:0'
                        ).get_shape().as_list())
                })

            if it_i % 10 == 0:
                print("iter:{} loss:{} content:{} style:{}".format(it_i,
                                                                   "{0:.2f}".format(loss.eval()),
                                                                   "{0:.2f}".format(content_loss.eval()),
                                                                   "{0:.2f}".format(style_loss.eval())))

            
            if it_i % gif_step == 0:
                imgs.append(np.clip(synth[0], 0, 1))

    return np.clip(synth, 0, 1)

def vgg_preprocess(x):
     return vgg16.preprocess(x, dsize=(224, 224))

import pdb
def style_in_batches(src="./raw/" , dest = "./stylized/", bs=50, start=0):
    
    fnames = list(os.listdir(src))
    max_files = len(fnames)
    num_batches = int(np.ceil( max_files / bs))
    
    style_img='./starry-night-art-plain.jpg'
    style_img = plt.imread(style_img)
    
    
    for i_batch in range(start, num_batches):
       
        print('batch {}/ {}'.format(i_batch + 1, num_batches))
        
        batch_fnames = fnames[i_batch * bs : min((i_batch + 1) * bs, max_files)]
        example = plt.imread('{}{}'.format(src,batch_fnames[0] ))
        imgs = np.zeros((len(batch_fnames),example.shape[0],example.shape[1],example.shape[2] ))
    
        for i in range(len(batch_fnames)):
            
            try:
                imgs[i] = plt.imread('{}{}'.format(src,batch_fnames[i] ))
            except:
                
                print('{}{}'.format(src,batch_fnames[i] ))
        
        synth = stylize(imgs, 
                style_img, 
                base_img=None, 
                saveto="../",
                gif_step=5, 
                n_iterations=50, 
                style_weight=2.0, 
                content_weight=1.0)
        
        
        
        for i in range(len(batch_fnames)):
            plt.imsave(fname='{}{}'.format(dest,batch_fnames[i]) ,arr=synth[i])
            
            
            
if __name__ == "__main__":   
    #Get raw viideo from S3 
    client = boto3.client('s3')
    client.download_file("artsnap-userfiles-mobilehub-1207532684","public/testvideo.mp4","testvideo.mp4")
    print("Done fetching raw file")

    ##Rip the frame and write to raw
    subprocess.call(["ffmpeg", "-i", "testvideo.mp4", "-r", "15", "-ss", "0:00", "-t", "20", "raw/out%d.png"])
    subprocess.call(["ffmpeg", "-i", "testvideo.mp4", "-ss", "0:00", "-t", "20", "-q:a", "0", "-map", "a", "testaudio.mp3" ])
    print("Done Rip the frame and audio. Start styling")

    #Styles frame images
    style_in_batches(src="./raw/" , dest = "./stylized/", bs=50, start=0)
    print("Done styling")


    # Assemble styled output to video and add back the audio
    # ffmpeg -framerate 15 -i stylized/out%d.png  -pix_fmt yuv420p yukirin_stylized.mp4
    subprocess.call(["ffmpeg", "-framerate", "15", "-i", "stylized/out%d.png",
                     "-pix_fmt", "yuv420p", "stylized_tmp.mp4"])
    #ffmpeg -i yukirin_stylized_s2.mp4 -i yukirin.mp3 -c:v copy -c:a aac -strict experimental yukirin_audio.mp4
    subprocess.call(["ffmpeg", "-i", "stylized_tmp.mp4", "-i", "testaudio.mp3", "-c:v",
                     "copy", "-c:a", "aac", "-strict", "experimental", "stylized.mp4"  ])
    print("Done recreate styled video")

    ## Upload video to s3
    client.upload_file("stylized.mp4","artsnap-userfiles-mobilehub-1207532684","public/stylized.mp4",ExtraArgs={'ACL':'public-read'})
    print("Done beaming video to s3")
    
    ## Clean up 
    subprocess.call(["rm", "testvideo.mp4"])
    subprocess.call(["rm", "stylized.mp4"])
    subprocess.call(["rm", "stylized_tmp.mp4"])
    subprocess.call(["rm", "testaudio.mp3"])
    files_raw = glob.glob("./raw/*")
    files_stylized = glob.glob("./stylized/*")
    files = files_raw + files_stylized
    for file in files:
        os.remove(file) 
    assert( len(list(os.listdir("raw"))) == 0)
    assert( len(list(os.listdir("stylized/"))) == 0)
   
    
        
        
    