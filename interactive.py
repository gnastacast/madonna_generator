import os
import tensorflow as tf
if tf.__version__[0] == '1':
    import tensorflow.contrib.slim as slim
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.contrib.slim as slim
import sys
import argparse
import importlib
import numpy as np
import h5py
import math
import time
from scipy import misc
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QGraphicsDropShadowEffect, QGraphicsOpacityEffect
from PyQt5.QtGui import QIcon, QPixmap, QTransform, QPainter, QImage, QRegion, QColor
from PyQt5.QtCore import QRect, Qt
from sklearn.neighbors import KDTree
from scipy.stats import norm
import cv2
import pickle

import threading

# import tensorflowjs as tfjs

class Decoder:
    def __init__(self, model_path, save=False):
        self.graph = tf.Graph()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0}), graph=self.graph)
        
        img_mean = np.array([134.10714722, 102.52040863, 87.15436554])
        img_stddev = np.sqrt(np.array([3941.30175781, 2856.94287109, 2519.35791016]))
        vae_def = importlib.import_module("src.generative.models.dfc_vae")
        vae = vae_def.Vae(100)
        gen_image_size = vae.get_image_size()
        
        with self.graph.as_default():
            tf.set_random_seed(666)
            latent_var = tf.placeholder(tf.float32, shape=(None,100), name="latent_var")

            # Create decoder
            reconstructed_norm = vae.decoder(latent_var, False)

            # Un-normalize
            self.reconstructed = (reconstructed_norm*img_stddev) + img_mean

            # Create a saver
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

            # Start running operations on the Graph
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            
            vae_checkpoint = os.path.expanduser(model_path)
            print('Restoring VAE checkpoint: %s' % vae_checkpoint)
            saver.restore(self.sess, vae_checkpoint)
            if save:
                tf.saved_model.simple_save(self.sess,
                                           'export_dir',
                                           inputs={"latent_var": latent_var},
                                           outputs={"reconstructed": self.reconstructed})
    
    def reconstruct(self, sweep_latent_var):
            recon = self.sess.run(self.reconstructed, feed_dict={"latent_var:0":sweep_latent_var})
            for im in recon:
                im[im<0] = 0
                im -= im.min()
                im[im>255] = 255
                im *= 1 / im.max()
            return recon
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
        
    def __enter__(self):
        return self
        
    def close(self):
        self.sess.close()

def sweepLatentVars(latent_vars, model_path, nrof_interp_steps = 10):
    with Decoder(model_path) as decoder:
        nrof_vars = len(latent_vars)
        sweep_latent_var = np.zeros((nrof_interp_steps*nrof_vars, 100), np.float32)
        for j in range(nrof_vars):
            first_var = latent_vars[j]
            second_var = latent_vars[(j + 1) % nrof_vars]
            for i in range(nrof_interp_steps):
                sweep_latent_var[i+nrof_interp_steps*j,:] = (1 - (i / nrof_interp_steps)) * first_var + \
                                                            (i / nrof_interp_steps) * second_var
        recon = decoder.reconstruct(sweep_latent_var)
        img = facenet.put_images_on_grid(recon, shape=(nrof_interp_steps*2,int(math.ceil(nrof_vars/2))))

        plt.figure(figsize = (16,10))
        plt.imshow(img)

def getLatentVars(latent_vars, model_path, indices):
    retval = [None] * len(indices)
    with Decoder(model_path) as decoder:
        retval = decoder.reconstruct(latent_vars)

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.fgLabel = QLabel(self)
        self.fgLabel.setAlignment(Qt.AlignCenter)
        self.mousePos = (.5, .5)
        filePath = os.path.dirname(os.path.abspath(__file__))
        facenetPath = os.path.join(filePath, 'facenet')
        with open(os.path.join(filePath, 'data', 'thumbs-clean', 'delaunay.pkl'), 'rb') as f:
            self.tsne, self.tsneRect, self.tri = pickle.load(f)
        with h5py.File(os.path.join(filePath, 'vae', '20180424-005429', 'attribute_vectors.h5'),'r') as f:
            self.latentVars = np.array(f.get('latent_vars'))
        self.tree = KDTree(self.tsneRect)
        modelPath = os.path.join(filePath, 'vae', '20180424-005429', 'model.ckpt-500')
        self.reconstructor = Decoder(modelPath)
        
        self.initUI()
        self.oldLatentVar= self.latentVars[0]
 
    def initUI(self):
        filePath = os.path.dirname(os.path.abspath(__file__))
        self.bgImage = cv2.flip(cv2.imread(os.path.join(filePath, "data", "thumbs-clean", "delaunay.png"), 1), 0)
        self.bgImage = cv2.cvtColor(self.bgImage.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # self.bgImage = cv2.imread("Thumbs-clean\\delaunay.png",0)
        # self.bgImage = self.bgImage - np.min(self.bgImage)
        # self.bgImage = self.bgImage / np.max(self.bgImage) * 255
        # self.bgImage = cv2.cvtColor(self.bgImage.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # bgMap = QPixmap("Thumbs-clean\\delaunay.png")
        # bgLabel = QLabel(self)
        # bgLabel.setPixmap(bgMap.scaled(self.height(), self.width()))

        # self.fgLabel = QLabel(self)
        # reconstruction = QPixmap("Thumbs-clean\\Virgin_Mary\\000.png")
        # self.fgLabel.setPixmap(reconstruction.scaled(self.height(), self.width()))

        self.dotSize = 10
        self.label = QLabel(self)
        self.label.resize(self.dotSize, self.dotSize)
        self.label.setStyleSheet('background-color: #FA8072; color: white; ')
        self.label.setMask(QRegion(self.label.rect(), QRegion.Ellipse))

        self.idleLabel = QLabel(self)
        self.idleLabel.resize(300, 300)
        self.idleLabel.setStyleSheet('background-color: #FA8072; font-size:32px; color: white')
        self.idleLabel.setText("Touch and drag\n to interact")
        self.idleLabel.setAlignment(Qt.AlignCenter)
        self.idleLabel.setMask(QRegion(self.idleLabel.rect(), QRegion.Ellipse))
        self.idleLabel.setAttribute(Qt.WA_TransparentForMouseEvents)
        dropShadow = QGraphicsDropShadowEffect()
        dropShadow.setBlurRadius(30)
        dropShadow.setOffset(0, 0)
        dropShadow.setColor(QColor(0,0,0))
        self.idleLabel.setGraphicsEffect(dropShadow)

        self.thread = threading.Thread()
        self.thread.run = self.print_time
        self.thread.daemon = True

        self.lock = threading.Lock()
        self.thread.start()

    def resizeEvent(self, event):
        QWidget.resizeEvent(self, event)
        self.fgLabel.setGeometry(0, 0, self.width(), self.height())
        

    def cleanUp(self):
        self.reconstructor.close()

    def getLatentVar(self):
        dataPtp = np.max(self.tsneRect, axis=0) - np.min(self.tsneRect, axis=0)
        dataMin = np.min(self.tsneRect, axis=0)
        
        p = np.array(self.mousePos) * dataPtp + dataMin
        nPoints = 10
        dist, idx = self.tree.query(p.reshape(1, -1), k=nPoints)
        deviation = np.sqrt(np.sum(dist) ** 2 / nPoints - 1)
        # print(deviation)
        distribution = norm(scale=deviation/5)
        weights = distribution.pdf(dist)
        if np.sum(weights):
            weights = weights / np.sum(weights)
        latentVar = np.zeros(self.latentVars.shape[1])
        for index, weight in zip(idx[0], weights[0]):
            # print("(%i, %.3f)" % (index, weight), end="")
            latentVar += self.latentVars[index] * weight

        return latentVar, deviation / np.max(dataPtp)

    # Define a function for the thread
    def print_time(self):
        count = 0
        idleCount = 0
        oldLatentVar = self.latentVars[0]
        while True:
            count += 1
            latentVar, maxDist = self.getLatentVar()
            if np.all(np.isclose(oldLatentVar, latentVar)):
                time.sleep(.01)
                idleCount += 1
                if idleCount > 100:
                    delay = 10
                    idlePos = min((idleCount - 100) / delay, 1) * np.pi / 2
                    idleSize = np.sin(idlePos) * 300
                    self.idleLabel.setGeometry((self.width() - self.height() - idleSize) / 2, (self.height() - idleSize) / 2, idleSize, idleSize)
                    self.idleLabel.setMask(QRegion(self.idleLabel.rect(), QRegion.Ellipse))
                    self.idleLabel.setVisible(True)
                continue
            self.idleLabel.setVisible(False)
            idleCount = 0
            oldLatentVar = latentVar
            cvImg = self.reconstructor.reconstruct([latentVar])[0]
            cvImg *= 255
            cvImg = cvImg.astype(np.uint8)
            cvImg = cv2.resize(cvImg, (self.fgLabel.height(), self.fgLabel.height()))#, interpolation = cv2.INTER_NEAREST)
            netImg = cv2.resize(self.bgImage, (self.fgLabel.width() - self.fgLabel.height(), self.fgLabel.height()))
            cvImg = np.hstack((netImg, cvImg))
            # # Create distance function for overlay
            # dist = np.ones((cvImg.shape[0], cvImg.shape[1]), dtype=np.uint8)
            # # Normalize mouse positions
            # x, y = np.clip(self.mousePos, 0, 0.999)
            # x, y = int(x * dist.shape[0]), int(y * dist.shape[1])
            # # Mark a zero for distance function
            # dist[y, x] = 0
            # dist = cv2.distanceTransform(dist, cv2.DIST_L2, 0)
            # fade = maxDist
            # dist = np.clip(fade - dist / np.max(dist), 0, 1) / fade
            # bgImage = cv2.resize(self.bgImage, cvImg.shape[0:2])
            # # cvImg = np.multiply(cvImg, np.multiply(dist[:,:,np.newaxis], 1 - bgImage / 255)).astype(np.uint8)
            # mask = np.multiply(dist[:,:,np.newaxis], 1 - bgImage / 255)
            # cvImg = np.multiply(cvImg, 1 - mask).astype(np.uint8)
            height, width, channel = cvImg.shape
            bytesPerLine = 3 * width
            qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
            reconstruction = QPixmap(qImg)
            self.lock.acquire()
            self.fgLabel.setPixmap(reconstruction)
            self.lock.release()
            time.sleep(.001)

    def mousePressEvent(self,event):
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.fgLabel.mapFromGlobal(event.globalPos())
        height, width = self.fgLabel.height(), self.fgLabel.width()
        x, y = pos.x() / self.fgLabel.height(), pos.y() / self.fgLabel.height()
        dividingLine = (width - height) / height
        if x < dividingLine:
           x = x / dividingLine
        else:
           x = x - dividingLine
        # self.mousePos = ((x - offsetX)/w, 1 - (y-offsetY)/h)
        self.mousePos = (x,y)
        dotSize = 10
        self.label.setGeometry(x * (width - height) - dotSize // 2, event.y() - dotSize // 2, dotSize, dotSize)
        # # self.label.setGeometry(w * self.mousePos[0] - dotSize // 2,  h * (1 - self.mousePos[1]) - dotSize // 2, dotSize, dotSize)
        self.repaint()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    filePath = os.path.dirname(os.path.abspath(__file__))
    facenetPath = os.path.join(filePath, 'facenet')
    sys.path.append(facenetPath)
    facenet = importlib.import_module("src.facenet")
    app = QApplication(sys.argv)
    QApplication.setOverrideCursor(Qt.BlankCursor);
    mw = QMainWindow()
    ex = App()
    mw.setCentralWidget(ex)
    app.aboutToQuit.connect(ex.cleanUp)
    # mw.resize(720,480)
    mw.showFullScreen()
    mw.setStyleSheet("background-color : black;")
    mw.show()
    sys.exit(app.exec_())
    # facenetPath = 'I:\\Work\\Art\\VirgenDeQuito\\GoogleSearch\\facenet'
    # sys.path.append('I:\\Work\\Art\\VirgenDeQuito\\GoogleSearch\\facenet')
    # facenet = importlib.import_module("src.facenet")
    # with h5py.File(os.path.expanduser(facenetPath + "\\vae\\20180424-005429\\attribute_vectors.h5"),'r') as f:
    #     latentVars = np.array(f.get('latent_vars'))



    # sweepLatentVars([latentVars[a] for a in [8,11,13,18,19,26,31,39,47,54,56,57,58,59,60,73, 75, 104, 112, 100]],
    #                  facenetPath + "\\vae\\20180424-005429\\model.ckpt-500")
