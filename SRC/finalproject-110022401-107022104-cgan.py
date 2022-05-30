# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# import packages

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import imageio
import shutil

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, LSTM, Permute, Reshape, Masking, TimeDistributed, MaxPooling1D, Flatten, Bidirectional
from keras.layers.merge import *
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import concatenate, maximum, dot, average, add, subtract
# from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers import Conv1D, GlobalMaxPooling1D, Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D
from keras.layers.merge import *
from keras.optimizers import *
from keras.regularizers import *
from keras.models import load_model

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.utils import shuffle

from scipy.stats import entropy
# -

pip install tables

# +
# # remove previous output

# shutil.rmtree('/kaggle/working/')

# +
# input files's path

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# +
# load Wprime

path_Wprime = '../input/phys591000-2022-final-project-iii/events_anomalydetection_DelphesPythia8_v2_Wprime_features.h5' 
Wprime_jet = pd.read_hdf(path_Wprime)

print('Wprime_jet.shape: ', Wprime_jet.shape)
print('Features: ', Wprime_jet.columns)

print('Wprime_jet')
display(Wprime_jet)

# +
# data processing: Wprime → training features + conditional label

pj1 = np.sqrt(Wprime_jet['pxj1']**2 + Wprime_jet['pyj1']**2 + Wprime_jet['pzj1']**2)
pTj1 = np.sqrt(Wprime_jet['pxj1']**2 + Wprime_jet['pyj1']**2)
phij1 = np.arccos(Wprime_jet['pxj1']/pTj1)
etaj1 = np.arcsinh(Wprime_jet['pzj1']/pTj1)
mj1 = Wprime_jet['mj1']
E1 = np.sqrt(pj1**2 + mj1**2)

pj2 = np.sqrt(Wprime_jet['pxj2']**2 + Wprime_jet['pyj2']**2 + Wprime_jet['pzj2']**2)
pTj2 = np.sqrt(Wprime_jet['pxj2']**2 + Wprime_jet['pyj2']**2)
phij2 = np.arccos(Wprime_jet['pxj2']/pTj2)
j2_rotate = phij2 - phij1
etaj2 = np.arcsinh(Wprime_jet['pzj2']/pTj2)
mj2 = Wprime_jet['mj2']
E2 = np.sqrt(pj2**2 + mj2**2)

mjj = np.sqrt((E1+E2)**2 - ((Wprime_jet['pxj1']+Wprime_jet['pxj2'])**2 + (Wprime_jet['pyj1']+Wprime_jet['pyj2'])**2 + (Wprime_jet['pzj1']+Wprime_jet['pzj2'])**2))

# +
# collect 7 training features

train = pd.DataFrame({'pTj1': pTj1,
                      'etaj1': etaj1,
                      'mj1': mj1,
                      'pTj2': pTj2,
                      'phij2': j2_rotate,
                      'etaj2': etaj2,
                      'mj2': mj2})

pd.reset_option('display')
print('train')
display(train)

# +
# rescale training features to [-1,1]

Train = train.values
scaler_Train = MinMaxScaler((-1, 1))
scaler_Train.fit(Train)
Train_rescaled = scaler_Train.transform(Train)

print('Train_rescaled.shape:', Train_rescaled.shape)

# +
# collect 1 training conditional label

condition_Train = pd.DataFrame({'mjj': mjj})

pd.reset_option('display')
print('condition_Train')
display(condition_Train)

# +
# rescale training conditional label to [0,1]

Condition_Train = condition_Train.values
scaler_Condition_Train = MinMaxScaler((0, 1))
scaler_Condition_Train.fit(Condition_Train)
Condition_Train_rescaled = scaler_Condition_Train.transform(Condition_Train)

print('Condition_Train_rescaled.shape:', Condition_Train_rescaled.shape)


# +
# generator of cGAN

def make_generator_cnn(GAN_noise_size, GAN_output_size):
    # Build Generative model ...

    G_input = Input(shape=(GAN_noise_size,))
    G_con_label = Input(shape=(1,))
    
    G_merge = Concatenate()([G_input, G_con_label])

    G = Dense(128, kernel_initializer='glorot_uniform')(G_merge)
    #G = Dropout(0.2)(G)
    G = LeakyReLU(alpha=0.2)(G)
    #G = Activation("relu")(G)
    G = BatchNormalization()(G)

    G = Reshape([8, 8, 2])(G)  # default: channel last

    G = Conv2DTranspose(32, kernel_size=2, strides=1, padding='same')(G)
    #G = Activation("relu")(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Conv2DTranspose(16, kernel_size=3, strides=1, padding='same')(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Flatten()(G)

    G_output = Dense(GAN_output_size)(G)
    G_output = Activation('tanh')(G_output)
    #G_output = Dense(GAN_output_size)(G)
    #G_output = LeakyReLU(0.2)(G_output)
    generator = Model([G_input, G_con_label], G_output)

    return generator


# +
# discriminator of cGAN

def make_discriminator_cnn(GAN_output_size):
    # Build Discriminative model ...
    # print "DEBUG: discriminator: input features:", GAN_output_size

    D_input = Input(shape=(GAN_output_size,))
    D_con_label = Input(shape=(1,))
    
    D_merge = Concatenate()([D_input, D_con_label])

    D = Dense(128)(D_merge)
    D = Reshape((8, 8, 2))(D)

    D = Conv2D(64, kernel_size=3, strides=1, padding='same')(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(32, kernel_size=3, strides=1, padding='same')(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(16, kernel_size=3, strides=1, padding='same')(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Flatten()(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Dropout(0.2)(D)

    D_output = Dense(1, activation='sigmoid')(D)
    #D_output = Dense(1)(D)

    discriminator = Model([D_input, D_con_label], D_output)
    
    return discriminator


# +
# build up cGAN models

tf.keras.backend.clear_session()

GAN_noise_size = 128
n_features = Train_rescaled.shape[1]

# d_optimizer = Adam(learning_rate=0.00001, beta_1=0.5, beta_2=0.9)
# g_optimizer = Adam(learning_rate=0.00001, beta_1=0.5, beta_2=0.9)

d_optimizer = SGD(0.01)
g_optimizer = SGD(0.01)

generator = make_generator_cnn(GAN_noise_size, n_features)
generator._name = "cGAN_Generator"
generator.compile(loss='mean_squared_error', optimizer=g_optimizer)
generator.summary()

discriminator = make_discriminator_cnn(n_features)
discriminator._name = "cGAN_Discriminator"
discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
discriminator.summary()

discriminator.trainable = False
GAN_input = Input(shape=(GAN_noise_size,))
GAN_con_label = Input(shape=(1,))
GAN_latent = generator([GAN_input, GAN_con_label])
GAN_output = discriminator([GAN_latent, GAN_con_label])
GAN = Model([GAN_input, GAN_con_label], GAN_output)
GAN._name = "cGAN"
GAN.compile(loss='binary_crossentropy', optimizer=g_optimizer)
GAN.summary()


# +
# build up training loop

def train_loop(epochs, batch_size):
    
    Train_rescaled_real_label = np.ones((batch_size, 1))
    Train_rescaled_fake_label = np.zeros((batch_size, 1))
    
    saved_epoch_list = []
    saved_fakedata_list = []
    epoch_g_stop = epochs
    epoch_d_stop = epochs
    d_acc_count = 0
    for epoch in range(epochs):
        
        if epoch < epoch_d_stop:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            Train_rescaled_idx = np.random.randint(0, Train_rescaled.shape[0], size=batch_size)
            Train_rescaled_real = Train_rescaled[Train_rescaled_idx, :]
            Condition_Train_rescaled_real = Condition_Train_rescaled[Train_rescaled_idx, :]

            # generate fake events
            Train_rescaled_noise = np.random.uniform(0, 1, size=[batch_size, GAN_noise_size])
            Train_rescaled_fake = generator.predict([Train_rescaled_noise, Condition_Train_rescaled_real])

            discriminator.trainable = True

            d_loss_r, d_acc_r = discriminator.train_on_batch([Train_rescaled_real, Condition_Train_rescaled_real], Train_rescaled_real_label)
            d_loss_f, d_acc_f = discriminator.train_on_batch([Train_rescaled_fake, Condition_Train_rescaled_real], Train_rescaled_fake_label)
            d_loss = 0.5 * np.add(d_loss_r, d_loss_f)
            d_acc = 0.5 * np.add(d_acc_r, d_acc_f)

            history['d_loss'].append(d_loss)
            history['d_loss_r'].append(d_loss_r)
            history['d_loss_f'].append(d_loss_f)
            history['d_acc'].append(d_acc)
            history['d_acc_r'].append(d_acc_r)
            history['d_acc_f'].append(d_acc_f)
            
            # accumulate continuous d_acc
            if d_acc == 1:
                d_acc_count += 1
            else:
                d_acc_count = 0
                
            # ---------------------
            #  Train Generator
            # ---------------------

            # we want discriminator to mistake images as real
            discriminator.trainable = False

            g_loss = GAN.train_on_batch([Train_rescaled_noise, Condition_Train_rescaled_real], Train_rescaled_real_label)
            history['g_loss'].append(g_loss)

            if epoch % 10000 == 0:
                print('Epoch: %d, discriminator(loss: %.3f, acc.: %.2f%%), generator(loss: %.3f)' % (epoch, d_loss, d_acc*100., g_loss))
                if epoch < epoch_g_stop:
                    saved_epoch_list.append(epoch)
                    saved_fakedata_list.append(scaler_Train.inverse_transform(generator([tf.random.uniform((10000, 128)), shuffle(Condition_Train_rescaled)[:10000]], training=False)))
            
            # d_loss < 0.5 and g_loss > 2 → save final useful info. of generator
            if d_loss < 0.5 and g_loss > 2 and epoch < epoch_g_stop:
                epoch_g_stop = epoch
                print('Generator training is good enough to stop!')
                print('Epoch: %d, discriminator(loss: %.3f, acc.: %.2f%%), generator(loss: %.3f)' % (epoch, d_loss, 100., g_loss))
                saved_epoch_list.append(epoch)
                saved_fakedata_list.append(scaler_Train.inverse_transform(generator([tf.random.uniform((10000, 128)), shuffle(Condition_Train_rescaled)[:10000]], training=False)))
                np.savez('cGAN_saved_fakedata_%d.npz' %(epoch_g_stop), epoch=saved_epoch_list, fakedata=saved_fakedata_list)
                generator.save('cGAN_generator_%d.h5' %(epoch_g_stop))
        
            # 100 times 100% → save final useful info. of discriminator
            if d_acc_count == 100:
                epoch_d_stop = epoch
                print('Discriminator training is good enough to stop!')
                print('Epoch: %d, discriminator(loss: %.3f, acc.: %.2f%%), generator(loss: %.3f)' % (epoch, d_loss, 100., g_loss))
                discriminator.save('cGAN_discriminator_%d.h5' %(epoch_d_stop))
                GAN.save('cGAN_%d.h5' %(epoch_d_stop))

    return epoch_g_stop, epoch_d_stop


# +
# train cGAN models & saved useful outputs

history = {'g_loss': [],
           'd_loss': [], 'd_loss_r': [], 'd_loss_f': [],
           'd_acc': [], 'd_acc_r': [], 'd_acc_f': []}

epochs = 250000
batch_size = 100
epoch_g_stop, epoch_d_stop = train_loop(epochs, batch_size)

with open('cGAN_history_%d.pickle' %(epoch_d_stop), 'wb') as f:
    pickle.dump(history, f)

# +
# plot final result & check K-L divergence

saved_generator = keras.models.load_model('cGAN_generator_%d.h5' %(epoch_g_stop))

realdata, realcondition = shuffle(Wprime_jet['mj1'], Condition_Train_rescaled)
fakedata = scaler_Train.inverse_transform(saved_generator([tf.random.uniform((10000, 128)), realcondition[:10000]], training=False))

realhist, realbins = np.histogram(realdata[:10000], bins = 25, range = (10, 600), density=1)
fakehist, fakebins = np.histogram(fakedata[:,2], bins = 25, range = (10, 600), density=1)

fig, axis = plt.subplots(1, 1, figsize=(8,8), dpi=150)
plt.title('epoch = '+str(epoch_g_stop), fontsize=20)
plt.ylim([0, 0.012])
plt.step(fakebins[:-1], fakehist, label = 'cGAN')
plt.step(realbins[:-1], realhist, label = 'Pythia8 Signal')
plt.xlabel('$m_{J_1}$', fontsize=20)
plt.legend(loc='upper right', fontsize=20)
plt.savefig('epoch_final.png')
plt.show()

def KL_divergent(p,q):
    return entropy(p,q)

print("KL Divergence D_KL(real||real): {:.3f}".format(KL_divergent(realhist[:], realhist[:])))
print("KL Divergence D_KL(fake||real): {:.3f}".format(KL_divergent(fakehist[:], realhist[:])))
print("KL Divergence D_KL(flat||real): {:.3f}".format(KL_divergent(np.full(len(realhist), 1/len(realhist)), realhist[:])))

# +
# plot loss function & accuracy

with open('cGAN_history_%d.pickle' %(epoch_d_stop), 'rb') as f:
    saved_history = pickle.load(f)

print(saved_history.keys())

plt.figure(figsize=(7,5), dpi=150)
plt.title('g_loss and d_loss')
plt.plot(saved_history['g_loss'], label='g_loss')
plt.plot(saved_history['d_loss'], label='d_loss')
plt.ylabel('loss function')
plt.xlabel('epoch')
plt.legend()
plt.savefig('loss.png')
plt.show()

plt.figure(figsize=(7,5), dpi=150)
plt.title('d_acc')
plt.plot(saved_history['d_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('accuracy.png')
plt.show()

# +
# plot .gif of epochs evolution

saved_fakedata = np.load('cGAN_saved_fakedata_%d.npz' %(epoch_g_stop))

print('files: ', saved_fakedata.files)
print('fakedata.shape: ', saved_fakedata['fakedata'].shape)

images = []
for i in range(saved_fakedata['fakedata'].shape[0]):
    
    fakedata_old = saved_fakedata['fakedata'][i]
    fakehist_old, fakebins_old = np.histogram(fakedata_old[:,2], bins = 25, range = (10, 600), density=1)
    
    fig, axis = plt.subplots(1, 1, figsize=(8,8), dpi=150)
    plt.step(fakebins_old[:-1], fakehist_old, label = 'cGAN')
    plt.step(realbins[:-1], realhist, label = "Pythia8 Signal")
    plt.title('epoch = '+str(saved_fakedata['epoch'][i]), fontsize=20)
    plt.ylim([0, 0.012])
    plt.xlabel('$m_{J_1}$', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig('epoch_'+str(i)+'.png')
    plt.close()

    images.append(imageio.imread('epoch_'+str(i)+'.png'))

imageio.mimsave('fakedata.gif', images, fps=1.3)

# +
# load test data

path_test_data = '../input/phys591000-2022-final-project-iii/test_sample_for_discriminator.npz'
test_data = np.load(path_test_data)

print('test_data.files: ', test_data.files)
print('test_sample.shape: ', test_data['test_sample'].shape)

test = pd.DataFrame(test_data['test_sample'])
test.columns = ['pxj1', 'pyj1', 'pzj1', 'mj1', 'pxj2', 'pyj2', 'pzj2', 'mj2']

pd.reset_option('display')
print('test')
display(test)

# +
# data processing: test data → testing features + conditional label

test_pj1 = np.sqrt(test['pxj1']**2 + test['pyj1']**2 + test['pzj1']**2)
test_pTj1 = np.sqrt(test['pxj1']**2 + test['pyj1']**2)
test_phij1 = np.arccos(test['pxj1']/test_pTj1)
test_etaj1 = np.arcsinh(test['pzj1']/test_pTj1)
test_mj1 = test['mj1']
test_E1 = np.sqrt(test_pj1**2 + test_mj1**2)

test_pj2 = np.sqrt(test['pxj2']**2 + test['pyj2']**2 + test['pzj2']**2)
test_pTj2 = np.sqrt(test['pxj2']**2 + test['pyj2']**2)
test_phij2 = np.arccos(test['pxj2']/test_pTj2)
test_j2_rotate = test_phij2 - test_phij1
test_etaj2 = np.arcsinh(test['pzj2']/test_pTj2)
test_mj2 = test['mj2']
test_E2 = np.sqrt(test_pj2**2 + test_mj2**2)

test_mjj = np.sqrt((test_E1+test_E2)**2 - ((test['pxj1']+test['pxj2'])**2 + (test['pyj1']+test['pyj2'])**2 + (test['pzj1']+test['pzj2'])**2))

# +
# collect 7 testing features

test_fit = pd.DataFrame({'pTj1': test_pTj1,
                         'etaj1': test_etaj1,
                         'mj1': test_mj1,
                         'pTj2': test_pTj2,
                         'phij2': test_j2_rotate,
                         'etaj2': test_etaj2,
                         'mj2': test_mj2})

pd.reset_option('display')
print('test_fit')
display(test_fit)

# +
# rescale testing features to [-1,1]

Test = test_fit.values
scaler_Test = MinMaxScaler((-1, 1))
scaler_Test.fit(Test)
Test_rescaled = scaler_Test.transform(Test)

print('Test_rescaled.shape:', Test_rescaled.shape)

# +
# collect 1 testing conditional label

condition_Test = pd.DataFrame({'mjj': test_mjj})

pd.reset_option('display')
print('condition_Test')
display(condition_Test)

# +
# rescale testing conditional label to [0,1]

Condition_Test = condition_Test.values
scaler_Condition_Test = MinMaxScaler((0, 1))
scaler_Condition_Test.fit(Condition_Test)
Condition_Test_rescaled = scaler_Condition_Test.transform(Condition_Test)

print('Condition_Test_rescaled.shape:', Condition_Test_rescaled.shape)

# +
# predict test data → result

saved_discriminator = keras.models.load_model('cGAN_discriminator_%d.h5' %(epoch_d_stop))

result = saved_discriminator.predict([Test_rescaled, Condition_Test_rescaled])
print('result.shape:', result.shape)

print('max of result:', np.max(result))
print('median of result:', np.median(result))
print('min of result:', np.min(result))

# +
# rescale result to [0,1]

scaler_result = MinMaxScaler((0, 1))
scaler_result.fit(result)
result_rescaled = scaler_result.transform(result)

print('result_rescaled.shape:', result_rescaled.shape)

# +
# plot distribution of rescaled result

result_rescaled_hist, result_rescaled_bins = np.histogram(result_rescaled, bins = 40, range = (0, 1), density=1)

fig, axis = plt.subplots(1, 1, figsize=(8,8), dpi=150)
plt.step(result_rescaled_bins[:-1], result_rescaled_hist, label = "Prediction")
plt.show()

print(np.max(result_rescaled))
print(np.median(result_rescaled))
print(np.min(result_rescaled))

# +
# load submission template

submission_template = pd.read_csv('../input/phys591000-2022-final-project-iii/submission_template_randomguess.csv')

pd.reset_option('display')
print('submission_template')
display(submission_template)

# +
# save submission prediciton

submission = submission_template.copy()

split_level = np.median(result_rescaled)
for i in range(result_rescaled.shape[0]):
    if result_rescaled[i] <= split_level:
        final_label = 0
    else:
        final_label = 1
    submission['prediction'][i] = final_label
    
pd.reset_option('display')
print('submission')
display(submission)

submission.to_csv('submission.csv', index=0)

# +
# check submission prediciton

count = 0
for i in submission['prediction']:
    if i == 1:
        count += 1
        
print('predict to real:', count)
print('predict to fake:', result_rescaled.shape[0] - count)
