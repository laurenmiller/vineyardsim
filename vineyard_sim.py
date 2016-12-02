
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from matplotlib.animation import FuncAnimation
#from mpl_toolkits.axes_grid1 import make_axes_locatable

def soil_variation(X):
    mux = 100*.75
    muy = 100*.75
    mux2 = 100*.24
    muy2=100*.4
    varx = 20
    vary = 20
    
    pd= np.exp(-((X[0] - .2*mux)**2/( 2*varx**2)) -
               ((X[1] - .2*muy)**2/(2*vary**2))) + \
               np.exp(-((X[0] - mux)**2/( 2*varx**2)) -
               ((X[1] - muy)**2/(2*vary**2))) + \
               np.exp(-((X[0] - mux2)**2/( 2*varx**2)) -
                      ((X[1] - muy2)**2/(2*vary**2)))
    return 2/1.*pd

class Plant(object):
	def __init__(self,position,init_soilmoisture):
		self.position=position
		self.soil_moisture=init_soilmoisture
		#self.drainage_rate=0
		#self.irrigation_rate=1
		#size of the leaves
		self.leaf_size=5 #size of each leaf for plotting
		self.leaf_num=5  #number of leaves to add each time the plant grows
		self.growth_ratio=2*np.array([[.1, 0], [0, .2]])
		self.leaf_positions = np.random.multivariate_normal(self.position, self.soil_moisture*self.growth_ratio, self.leaf_num)
		self.color()

	def grow(self,irrigation_rate,drainage_rate):
		self.soil_moisture=self.soil_moisture+irrigation_rate-drainage_rate
		if self.soil_moisture>=0:
			newleaves = np.random.multivariate_normal(self.position, self.soil_moisture*self.growth_ratio, self.leaf_num)
			self.leaf_positions=np.vstack((self.leaf_positions,newleaves))
		else:
			self.soil_moisture=0
		self.color()
		

	def color(self):
		nlvs=len(self.leaf_positions[:, 0])

		if self.soil_moisture>0:
			colors= [0,153/256.0,0,.2]+np.random.uniform(-.2, .2, (nlvs,4))+[-10/256.0,0,0,0]
		else:
			colors=[128/256.0, 128/256.0, 0/256.0,1]+np.random.uniform(-.1, .1, (nlvs,4))
		colors=colors.tolist()
		self.colors= np.clip(colors, 0, 1)

class Vineyard(object):
	def __init__(self):
		# Create new Figure and an Axes which fills it.
		self.bounds=[[0,100],[0,100]]
		self.fig = plt.figure(figsize=(10, 5))
		gs = gridspec.GridSpec(1, 2,width_ratios=[1,1.1])
		self.ax1 = plt.subplot(gs[0])
		self.ax2 = plt.subplot(gs[1])
		#self.ax1=self.fig.add_subplot(121)
		#self.ax2=self.fig.add_subplot(122)
		self.ax1.set_aspect('equal')
		self.ax2.set_aspect('equal')
		#ax1 = fig.add_axes([0, 0, 1, 1], frameon=True)
		self.ax1.set_xlim(-10, self.bounds[0][1]+10), self.ax1.set_xticks([])
		self.ax1.set_ylim(-10, self.bounds[1][1]+10), self.ax1.set_yticks([])
		self.ax2.set_xlim(-10, self.bounds[0][1]+10), self.ax2.set_xticks([])
		self.ax2.set_ylim(-10, self.bounds[1][1]+10), self.ax2.set_yticks([])
		self.ax1.set_axis_bgcolor((152/256.0, 107/256.0, 73/256.0))

		# Initialize the plants on a grid
		nx, ny = (10, 20)
		x = np.linspace(0, self.bounds[0][1], nx)
		y = np.linspace(0, self.bounds[1][1], ny)
		xx, yy = np.meshgrid(x, y)

		self.vine_positions = np.vstack((xx.flatten(),yy.flatten())).T

		#each vine as it's own drainage rate
		self.drainage_rate = soil_variation(self.vine_positions.T)
		
		#each vine as it's own irrigation rate
		
		#constant irrigation
		#self.irrigation_rate = 2*np.ones(self.vine_positions.shape[0])
		
		#optimal irrigation
		self.irrigation_rate =1*np.ones(self.vine_positions.shape[0])+self.drainage_rate
		
		#initialize the starting oil moisture level
		init_soilmoisture=1

		#create vines
		self.vines=[]
		for pos in self.vine_positions:
			self.vines.append(Plant(pos,init_soilmoisture))

		#set up some plotting stuff
		dr= self.drainage_rate.reshape(20,10)
		sc=self.ax2.imshow(np.flipud(dr),cmap=plt.get_cmap('BrBG_r'),extent=(0, self.bounds[0][1],0, self.bounds[1][1]))
		#legend
		df=self.fig.colorbar(sc,fraction=0.046, pad=0.04)
		df.ax.set_yticklabels(['slow','','','','','','','fast'])
		df.set_label('soil drainage', rotation=270)

		#plt.tight_layout()
		self.ind=0
		

	
	def update(self,i):
		# self.leaf_num=self.leaf_num+5
		#self.soil_moisture=self.soil_moisture-self.drainage_rate+self.irrigation_rate

		#leafpositions=[]
		sizes=[]
		moistures=[]
		for ind,vine in enumerate(self.vines):
			vine.grow(self.irrigation_rate[ind],self.drainage_rate[ind])

			# plot stuff	
			if ind==0:

				leafpositions=vine.leaf_positions
				colors=vine.colors
				vinepositions=vine.position
			else:
				leafpositions=np.vstack((leafpositions,vine.leaf_positions))
				colors=np.vstack((colors,vine.colors))
				vinepositions=np.vstack((vinepositions,vine.position))

			sizes.append(vine.leaf_size)
			moistures.append(vine.soil_moisture)
		#print leafpositions.shape
		scat1 = self.ax1.scatter(leafpositions[:, 0], leafpositions[:, 1],
		                  	s=sizes,  edgecolors=colors,
		                  	facecolors=colors
		                  	)
		scat2 = self.ax2.scatter(vinepositions[:,0],vinepositions[:,1],
		                  	s=5*self.irrigation_rate,  edgecolors='blue',
		                  	facecolors='blue'
		                  	)

		self.ind=self.ind+1
		#pt.savefig('img'+str(self.ind)+'.png',dpi=1000)

	def animate(self):
		anim=FuncAnimation(self.fig, self.update)
		anim.save('optimal_irrigation.mp4', fps=20,bitrate=100000)
		print 'done saving'
		#plt.show()

vy=Vineyard()
vy.animate()
#animation = FuncAnimation(fig, vy.update,frames=5, interval=20)
# #animation.save('basic_animation2.mp4', fps=50,bitrate=5000)
#plt.show()
