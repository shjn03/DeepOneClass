import tensorflow as tf
from keras import layers
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD
from keras.applications import MobileNetV2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
from keras.datasets import fashion_mnist
import numpy as np
from tqdm import tqdm
from keras.models import Model,load_model
from keras.layers import Input,Lambda,Dense,Dropout,Flatten
from sklearn.metrics import euclidean_distances, roc_auc_score
import cv2
import bhtsne
import sklearn
import sklearn.base
import scipy as sp
import keras
from sklearn import metrics
from cnn_model_builder import seresnet_v2,build_cnn
import matplotlib.pyplot as plt
import time
import datetime
import subprocess
from keras.regularizers import l2
import pandas as pd
#################################################
#Use GPU as much as necessary(tensorflow only)
from keras import backend as K
if K._BACKEND=="tensorflow":
    from keras.backend import tensorflow_backend
    
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

###################################################
#Create output folder
fname,ext=os.path.splitext(os.path.basename(__file__))
todaydetail  =    datetime.datetime.today()
outputdir=os.path.join("log",todaydetail.strftime("%Y%m%d_%H%M%S_"+fname))
os.makedirs(outputdir)

###################################################
#saving script log 
#log=commands.getoutput("cat "+os.path.basename(__file__))#input filename
log=subprocess.check_output(["cat",os.path.basename(__file__)])#input filename)    
with open(outputdir +'/script_log.py', 'w') as f: #saving logfile
    f.write(log.decode("utf-8")) 





def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def createModel(base_model_path=None,emb_size=128,input_shape=(28,28,1)):

    # Initialize a ResNet50_ImageNet Model
   
    #resnet_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)
    if base_model_path is not None:
        print ("loading pretrained_model :{}".format(base_model_name))
        base_model=load_model(base_model_path)
        x = base_model.layers[-2].output
    else:
        print ("build and initalize model :{}".format("ResNet50"))
#        base_model=keras.applications.resnet50.ResNet50(weights=None, 
#                                                        include_top = False, 
                                              #input_tensor=Input(shape=input_shape))
        n=2
        depth = n * 9 + 2
        #base_model=seresnet_v2(input_shape, depth, num_classes=10)
        base_model=build_cnn(input_shape,num_classes=2)
        x = base_model.layers[-2].output
    
    
    embedded = Dense(emb_size,kernel_regularizer=l2(1e-4))(x)
    # New Layers over ResNet50
#    net = resnet_model.output
    #net = kl.Flatten(name='flatten')(net)
#    net = kl.GlobalAveragePooling2D(name='gap')(net)
    #x = Dropout(0.5)(x)
#    net = kl.Dense(emb_size,activation='relu',name='t_emb_1')(net)
    #embedded = Lambda(lambda  x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(x)

    # model creation
    siamese_model = Model(base_model.input, embedded, name="base_model")

    # triplet framework, shared weights
    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_pos')
    input_negative = Input(shape=input_shape, name='input_neg')

    net_anchor = siamese_model(input_anchor)
    net_positive = siamese_model(input_positive)
    net_negative = siamese_model(input_negative)

    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
    tertiary_dist = Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])

    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

    # Setting up optimizer designed for variable learning rate

    # Variable Learning Rate per Layers
    #lr_mult_dict = {}
    if base_model_path is not None:
        last_freeze_layer = model.layers[-4].name
        for layer in model.layers:
            # comment this out to refine earlier layers
            if layer.name==last_freeze_layer:
                break;
            layer.trainable = False  
        # print layer.name
#        lr_mult_dict[layer.name] = 1
        # last_layer = layer.name
#    lr_mult_dict['t_emb_1'] = 100
#
#    base_lr = 0.0001
#    momentum = 0.9
#    v_optimizer = LR_SGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, multipliers = lr_mult_dict)

    model.compile(optimizer="adam", loss=triplet_loss, metrics=[accuracy])

    return model


# OnlineのTriplet選択
def triplet_loss_v2(label, embeddings):
    # バッチ内のユークリッド距離
    x1 = tf.expand_dims(embeddings, axis=0)
    x2 = tf.expand_dims(embeddings, axis=1)
    euclidean = tf.reduce_sum((x1-x2)**2, axis=-1)

    # ラベルが等しいかの行列（labelの次元が128次元になるので[0]だけ取る）
    lb1 = tf.expand_dims(label[:, 0], axis=0)
    lb2 = tf.expand_dims(label[:, 0], axis=1)
    equal_mat = tf.equal(lb1, lb2)

    # positives
    positive_ind = tf.where(equal_mat)
    positive_dists = tf.gather_nd(euclidean, positive_ind)

    # negatives
    negative_ind = tf.where(tf.logical_not(equal_mat))
    negative_dists = tf.gather_nd(euclidean, negative_ind)

    # [P, N]
    positives = tf.expand_dims(positive_dists, axis=1)
    negatives = tf.expand_dims(negative_dists, axis=0)
    triplets = tf.maximum(positives - negatives + 0.2, 0.0) # Margin=0.2
    return tf.reduce_mean(triplets) # sumだと大きなりすぎてinfになるため

def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - 0.5*(K.square(y_pred[:,1,0])+K.square(y_pred[:,2,0])) + margin))

import Augmentor
def get_Augmentor(path,batch_size,rotate=True,color=True,flip_ud=True):
    p = Augmentor.Pipeline(path, output_directory="/tmp/")
    p.skew_tilt(probability=0.5, magnitude=0.5)
    p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=2)
    if rotate:
        p.rotate90(probability=0.3)
        p.rotate270(probability=0.3)
        p.rotate(probability=0.3, max_left_rotation=10, max_right_rotation=10)
    if flip_ud:
        p.flip_top_bottom(probability=0.3)
    p.flip_left_right(probability=0.3)

    if not color:
        p.greyscale(probability=1.0)
    g=p.keras_generator(batch_size=batch_size)
    return g
    
def generator_from_Augmentor(normal_g,anomaly_g):
    while True:
        num_samples=10
        for i in range(num_samples//batch_size):
            x_anchors,_=next(normal_g)
            x_positives,_=next(normal_g)
            x_negatives,_=next(anomaly_g)
            y=np.random.randint(2, size=(1,2,batch_size)).T#dummy
            yield [x_anchors,x_positives,x_negatives],y
            
def load_data(path,color=False):
    files=os.listdir(path)
    if os.path.isdir(os.path.join(path,files[0])):
        file_num=0 
        for dirname in files:
             imgs=os.listdir(os.path.join(path,dirname))
             tmp=cv2.imread(os.path.join(path,dirname,imgs[0]))
             #import pdb;pdb.set_trace()
             height,width,_=tmp.shape
             file_num+=len(imgs)
        x_sample=np.zeros((file_num,height,width,3 if color else 1),dtype=np.float32)
        file_count=0
        for dirname in files:
            imagenames=os.listdir(os.path.join(path,dirname))
            for i,imagename in enumerate(imagenames):
                img=cv2.imread(os.path.join(path,dirname,imagename))
                if color:
                    x_sample[file_count,:,:]=img
                else:
                    x_sample[file_count,:,:,0]=img[:,:,0]
                file_count+=1
        x_sample/=255
        
    else:
        #single dir mode
        imagenames=files
        tmp=cv2.imread(os.path.join(path,imagenames[0]))
        height,width,_=tmp.shape
        x_sample=np.zeros((len(files),height,width,3 if color else 1),dtype=np.float32)
        for i,filename in enumerate(files):
            img=cv2.imread(os.path.join(path,filename))
            if color:
                x_sample[i,:,:]=img
            else:
                img=img[:,:,0]
                x_sample[i,:,:,0]=img
        x_sample/=255
    return x_sample,imagenames
    
    
    

def step_decay(epoch):
    x = 1e-3
    if epoch >= 60: x /= 10.0
    if epoch >= 100: x /= 10.0
    if epoch >= 200: x /= 10.0
    return x


def train_generator(X, y_label, batch_size):
    while True:
        indices = np.random.permutation(X.shape[0])
        for i in range(len(indices)//batch_size):
            current_indices = indices[i*batch_size:(i+1)*batch_size]
            X_batch = X[current_indices] / 255.0
            y_batch = np.zeros((batch_size, 128), np.float32)
            y_batch[:,0] = y_label[current_indices]
            yield X_batch, y_batch

def train_pairs_generator(x_normal, x_anomaly, batch_size):
    while True:
        num_normals=x_normal.shape[0]
        num_anomalys=x_anomaly.shape[0]
        if num_normals>num_anomalys*2:
            num_pairs=num_anomalys#*num_normals
        else:
            num_pairs=num_normals//2
        idx_normals=np.random.permutation(num_normals)
        idx_anomalys=np.random.permutation(num_anomalys)
        idx_anchors=idx_normals[:num_pairs]
        idx_positives=idx_normals[num_pairs:num_pairs*2]
        idx_negatives=idx_anomalys[:num_pairs]
        for i in range(num_pairs//batch_size):
            idx_start=i*batch_size
            idx_end=i*batch_size+batch_size
            x_anchors=x_normal[idx_anchors[idx_start:idx_end]]
            x_positives=x_normal[idx_positives[idx_start:idx_end]]
            x_negatives=x_anomaly[idx_negatives[idx_start:idx_end]]
            y=np.random.randint(2, size=(1,2,batch_size)).T#dummy
            yield [x_anchors,x_positives,x_negatives],y

class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed

    def fit_transform(self, x):
        return bhtsne.tsne(
            x.astype(np.float64), dimensions=self.dimensions, perplexity=self.perplexity, theta=self.theta,
            rand_seed=self.rand_seed)


if __name__ == "__main__":
    base_path="data/tak/cond1_N_anomaly_10"
    train_path=os.path.join(base_path,"train")
    test_path=os.path.join(base_path,"test")
    valid_path=os.path.join(base_path,"valid")
    base_model_name="best_model_at_epoch-194-val_acc_0.817.h5"
    batch_size=10
    emb_size=128 
    epochs=200
    color=False
    Use_Augmentor=True
    Use_Pretrained_model=False
    if Use_Augmentor:
        normal_g=get_Augmentor(os.path.join(base_path,"train","normal"),batch_size=batch_size)
        anomaly_g=get_Augmentor(os.path.join(base_path,"train","anomaly"),batch_size=batch_size)
        trainGene=generator_from_Augmentor(normal_g,anomaly_g)
        x_train_normal,tr_n_filenames=load_data(os.path.join(train_path,"normal"),color=color)
        x_valid_normal,vl_n_filenames=load_data(os.path.join(valid_path,"normal"),color=color)
        x_test_normal,ts_n_filenames=load_data(os.path.join(test_path,"normal"),color=color)
        x_test_anomaly,ts_a_filenames=load_data(os.path.join(test_path,"anomaly"),color=color)
        _,input_height,input_width,input_channels=x_test_normal.shape
    else:
        x_train_normal,tr_n_filenames=load_data(os.path.join(train_path,"normal"),color=color)
        x_train_anomaly,tr_a_filenames=load_data(os.path.join(train_path,"anomaly"),color=color)
        x_test_normal,ts_n_filenames=load_data(os.path.join(test_path,"normal"),color=color)
        x_test_anomal,ts_a_filenames=load_data(os.path.join(test_path,"anomaly"),color=color)
        _,input_height,input_width,input_channels=x_train_normal.shape
        trainGene=train_pairs_generator(x_train_normal,x_train_anomaly,batch_size=batch_size)

    test_num_normals=x_test_normal.shape[0]
    test_num_anomalys=x_test_anomaly.shape[0]
    normal_ratio= test_num_normals/(test_num_normals+test_num_anomalys)

    model=createModel(base_model_path=base_model_name if Use_Pretrained_model else None,emb_size=emb_size,input_shape=(input_height,input_width,input_channels))
    pred_model = model.get_layer('base_model')
    print("Done.")

    
    scheduler = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint("best_model.h5", monitor="loss", save_best_only=True)
    callbacks=[scheduler,checkpoint]
    num_samples=x_train_normal.shape[0]
    best_f1=0
    best_model_name = "best_model_f1_{}.h5".format(best_f1)
    best_model_path = os.path.join(outputdir, best_model_name)
    hist_columns=["train_loss","train_accuracy","val_accuracy","val_precision","val_recall","val_f1_score"]
    df_history=pd.DataFrame(columns=hist_columns)
    for e in range(epochs):
        print("epoch:{}/{}".format(e+1,epochs))
        history=model.fit_generator(trainGene, 
                            steps_per_epoch=num_samples//(2*batch_size), 
                            epochs=1, 
                            shuffle=True, 
                            use_multiprocessing=True,
                            #callbacks=[checkpoint]
                            )
        train_acc=history.history["accuracy"][0]
        train_loss=history.history["loss"][0]
        
        print("evaluating...")
        pred_normal=pred_model.predict(x_test_normal)
        pred_anomaly=pred_model.predict(x_test_anomaly)
        pred_anchors=pred_model.predict(x_valid_normal)
        test_num_normals=pred_normal.shape[0]
        test_num_anomalys=pred_anomaly.shape[0]
        
        preds=np.r_[pred_normal,pred_anomaly]
        dist_matrix = np.zeros((preds.shape[0], pred_anchors.shape[0]), np.float32)
        for i in range(dist_matrix.shape[0]):
            dist_matrix[i,:] = euclidean_distances(preds[i,:].reshape(1,-1),
                                               pred_anchors)[0]
        min_dists = np.min(dist_matrix, axis=-1)
        th=np.percentile(min_dists,normal_ratio*100)
        y_pred=(min_dists>th).astype(np.uint8)
        y_test=np.zeros(preds.shape[0])
        y_test[test_num_normals:]=1#for calc auc normal=0,anomaly=1
        acc=metrics.accuracy_score(y_test, y_pred)
        prc=metrics.precision_score(y_test, y_pred)
        rcl=metrics.recall_score(y_test, y_pred)
        f1=metrics.f1_score(y_test, y_pred)
        
        series_temp=pd.Series([train_loss,train_acc,acc,prc,rcl,f1],index=hist_columns)
        df_history=df_history.append(series_temp,ignore_index=True)
        
        print("val_accuacy : {:.3} , val_precision : {:.3} , val_recall : {:.3} , val_f : {:.3}".format(acc,prc,rcl,f1))
        if f1>best_f1:
            print("f1 improved from {:.3}".format(f1))
            best_f1=f1
        model.save(os.path.join(outputdir,"last_model.h5".format(best_f1)))
    try:
        #print("Loadding best f1 model:{}".format(base_model))
        model=load_model("best_model_f1_{}.h5".format(best_f1),custom_objects={'triplet_loss':triplet_loss })
    except:
        print("failed to load_best_model.use last weights")
    #%%
    plt.figure()
    df_history.plot(legend=True)
    plt.savefig(os.path.join(outputdir,"history.png"))
    
    pred_normal=pred_model.predict(x_test_normal,verbose=1)
    pred_anomaly=pred_model.predict(x_test_anomaly,verbose=1)


    preds=np.r_[pred_normal,pred_anomaly]    
    y_test=np.zeros(preds.shape[0])
    y_test[test_num_normals:]=1#for calc auc normal=0,anomaly=1
    
    pred_anchors=pred_model.predict(x_valid_normal)
    
    dist_matrix = np.zeros((preds.shape[0], pred_anchors.shape[0]), np.float32)
    for i in range(dist_matrix.shape[0]):
        dist_matrix[i,:] = euclidean_distances(preds[i,:].reshape(1,-1),
                                           pred_anchors)[0]
    min_dists = np.min(dist_matrix, axis=-1)
    plt.figure()
    plt.title("histgram normal vs anomaly")
    plt.hist(min_dists[:pred_normal.shape[0]],color="b",bins=100,alpha=0.8,label="normal")
    plt.hist(min_dists[pred_normal.shape[0]:],color="r",bins=100,alpha=0.8,label="anomaly")
    plt.legend()
    plt.savefig(os.path.join(outputdir,"histgram.png"))
    
    auc=roc_auc_score(y_test, min_dists)
    fpr, tpr, _ = metrics.roc_curve(y_test, min_dists)
    # AUC
    auc = metrics.auc(fpr, tpr)
    
    # ROC曲線をプロット
    plt.figure()
    plt.plot(fpr, tpr, label='DeepOneClassification(AUC = %.4f)'%auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(outputdir,"ROC curve.png"))
    
    #T-SNE
    start=time.time()
    BH_TSNE = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1)
    x_embeded = BH_TSNE.fit_transform(preds)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    
    num_classes=2
    plt.figure(figsize=(10,10))
    plt.scatter(x_embeded[:test_num_normals,0],x_embeded[:test_num_normals,1],label="normal",c="b",alpha=0.5)
    plt.scatter(x_embeded[test_num_normals:,0],x_embeded[test_num_normals:,1],label="anomaly",c="r",alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(outputdir,"tsne.png"))
    
    df=pd.DataFrame(min_dists,)
    
