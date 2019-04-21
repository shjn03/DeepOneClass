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
from keras.layers import Input,Lambda,Dense
from sklearn.metrics import euclidean_distances, roc_auc_score
import cv2
import bhtsne
import sklearn
import sklearn.base
import scipy as sp
import keras
from sklearn import metrics
from cnn_model_builder import seresnet_v2
import matplotlib.pyplot as plt
import time
import datetime
import subprocess
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
        base_model=seresnet_v2(input_shape, depth, num_classes=10)
        x = base_model.layers[-2].output
    
    
    embedded = Dense(emb_size)(x)
    # New Layers over ResNet50
#    net = resnet_model.output
    #net = kl.Flatten(name='flatten')(net)
#    net = kl.GlobalAveragePooling2D(name='gap')(net)
#    #net = kl.Dropout(0.5)(net)
#    net = kl.Dense(emb_size,activation='relu',name='t_emb_1')(net)
#    net = kl.Lambda(lambda  x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(net)

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
        last_freeze_layer = model.layers[-8].name
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


def load_data(path):
    files=os.listdir(os.path.join(path,"normal"))    
    tmp=cv2.imread(os.path.join(path,"normal",files[0]))
    height,width,color=tmp.shape
    x_normal=np.zeros((len(files),height,width,color),dtype=np.float32)
    for i,filename in enumerate(files):
        img=cv2.imread(os.path.join(path,"normal",filename))
        img=img[:,:,0]
        x_normal[i,:,:,0]=img
    x_normal/=255
    files=os.listdir(os.path.join(path,"anomaly"))
    x_anomaly=np.zeros((len(files),height,width,color),dtype=np.float32)
    for i,filename in enumerate(files):
        img=cv2.imread(os.path.join(path,"anomaly",filename))
        img=img[:,:,0]
        x_anomaly[i,:,:,0]=img
    x_anomaly/=255
    return x_normal,x_anomaly
    
    
    

def step_decay(epoch):
    x = 1e-3
    if epoch >= 25: x /= 10.0
    if epoch >= 45: x /= 10.0
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
    base_path="data/dog_vs_cat(all)"
    train_path=os.path.join(base_path,"train")
    test_path=os.path.join(base_path,"test")
    base_model_name="best_mnist_model_at_epoch-88-val_acc_0.995.h5"
    batch_size=16
    x_train_normal,x_train_anomaly=load_data(train_path)
    x_test_normal,x_test_anomaly=load_data(test_path)
    _,input_height,input_width,input_channels=x_train_normal.shape
    

    model=createModel(base_model_path=None,input_shape=(input_height,input_width,input_channels))
    print("Done.")
    trainGene=train_pairs_generator(x_train_normal,x_train_anomaly,batch_size=batch_size)
    epochs=100
    
    scheduler = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint("best_model.h5", monitor="loss", save_best_only=True)
    
    best_model_name = "best_model_at_epoch-{epoch:02d}-acc_{acc:.3f}.h5"
    best_model_path = os.path.join(outputdir, best_model_name)
    for e in range(epochs):
        print("epoch:{}/{}".format(e+1,epochs))
        model.fit_generator(trainGene, 
                            steps_per_epoch=len(x_train_normal//2) / batch_size, 
                            epochs=1, 
                            shuffle=True, 
                            use_multiprocessing=True,
                            callbacks=[checkpoint])
        model.predict
    try:
        model=load_model(best_model_path)
    except:
        print("failed to load_best_model.use last weights")
    #%%
    pred_model = model.get_layer('base_model')
    pred_normal=pred_model.predict(x_test_normal,verbose=1)
    pred_anomaly=pred_model.predict(x_test_anomaly,verbose=1)
    test_num_normals=pred_normal.shape[0]
    test_num_anomalys=pred_anomaly.shape[0]

    preds=np.r_[pred_normal,pred_anomaly]    
    y_test=np.zeros(preds.shape[0])
    y_test[test_num_normals:]=1#for calc auc normal=0,anomaly=1
    
    pred_anchors=pred_model.predict(x_train_normal)
    
    dist_matrix = np.zeros((preds.shape[0], pred_anchors.shape[0]), np.float32)
    for i in range(dist_matrix.shape[0]):
        dist_matrix[i,:] = euclidean_distances(preds[i,:].reshape(1,-1),
                                           pred_anchors)[0]
    min_dists = np.min(dist_matrix, axis=-1)
    plt.figure()
    plt.title("histgram normal vs anomaly")
    plt.hist(min_dists[:pred_normal.shape[0]],color="b",bins=100,alpha=0.8)
    plt.hist(min_dists[pred_normal.shape[0]:],color="r",bins=100,alpha=0.8)
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
    plt.show()
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
