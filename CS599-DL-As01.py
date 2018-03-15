import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import numpy as np 

from tensorflow.example.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
num_sample = mnist.train_numexamples
input_dim = 784   # 28*2*
w = h = 28

class VariationalAuencoder(object):
	def __init__ (self, learning_rate=1e-4, batch_size=100, n_z=5):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.n_z = n_z
		# build the network
		self.build()
		# lunch a session

	#build the network
	def build(self):
		#input
		self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])
		
		#encoder
		#slim.fc(input, output_dim, scope, act_fn)
		f1= fc(self.x, 512, scope='enc_fc1', activation_fn = tf.nn.elu)
		f2= fc(f1, 384, scope='enc_fc2', activation_fn = tf.nn.elu)
		f3= fc(f2, 256, scope='enc_fc3', activation_fn = tf.nn.elu)

		self.z_mu = fc(f3,self.n_z, scope='enc_fc4_mu', activation_fn=None)
		#log(signma^2)
		self.z_log_sigma_sq = fc(f3,self.n_z, scope='enc_fc4_mu', activation_fn=None)

		#N(z_mu, z_sigma)
		eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)

		self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
		#decoder
		g1 = fc(self.z, 256, scope='dec_fc1', activation_fn=tf.nn.elu)
		g2 = fc(g1, 384, scope='dec_fc2', activation_fn=tf.nn.elu)
		g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.elu)
		self.x_hat = fc(g3, input_dim, scope='dec_fc4', activation_fn=tf.sigmoid)


		#losses
		#reconstruction loss
		#x <--> x_hat
		#H(x,x_hat) = - \Sigma x * log(x_hat) + (1-x) * log(1-x_hat)
		epsilon = 1e-10
		recon_loss = -tf.reduce_sum(self.x * tf.log(self.x_hat + epsilon) + (1 - self.x)*tf.log(1- self.x_hat + epsilon) )

		#latest loss
		#KL divergence: measure the different between two distributions
		#the latest distribution and N(0,1)
		latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq),axis=1)

		#total loss
		
		#optimizer
	# execute a forward and a backward pass
	def run_single_step():

	#reconstruction
	def reconstructor():

	#generation
	def generator():

	#transformation
	def transformer():

def trainer():
	# model
	# training loop

