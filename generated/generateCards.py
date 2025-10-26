import glob
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import cv2



margin = 0.000
mary_data = []

width = 2
height = width * 2
margin = width / 5

def drawCard(img, text, center, name):

	def label(xy, text):
	    y = xy[1] - height/3.3  # shift y-value for label so that it's below the artist
	    ax.text(xy[0], y, text, ha="center", va='center', family='serif', size=6, fontname='AR PL UKai CN')
	fig = plt.figure(figsize=(4,8),dpi=150)
	ax = fig.add_subplot(1,1,1)
	# add a fancy box
	fancybox = mpatches.FancyBboxPatch(
	    center - [width/2, height/2], width, height,
	    boxstyle=mpatches.BoxStyle("Round", pad=margin + margin/5),
	    edgecolor=(0,0,0),
	    facecolor=(1,1,1,0),
	    linestyle="dotted",
	)
	ax.add_patch(fancybox)
	# add a fancy box
	fancybox = mpatches.FancyBboxPatch(
	    center - [width/2, height/2], width, height,
	    boxstyle=mpatches.BoxStyle("Round", pad=margin),
	    edgecolor=(.8,.8,.8),
	    facecolor=(1,1,1,0),
	    linestyle="solid",
	)
	ax.add_patch(fancybox)
	label(center, text)
	ax.imshow(
		img,
		extent= np.hstack([
			center + [-width/2, width/2],
			center + [0.00, width],
		]),
	)


	plt.axis('equal')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(name, format="pdf")
	plt.close()

with open("mary_metadata.csv", 'r') as f:
	for i, line in enumerate(f.readlines()):
		line = line.split(',')
		center = np.array([0,0])
		img = cv2.imread(line[0])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		text = ""
		for j, v in enumerate(line[1:]):
			if j % 5 == 0:
				text += "\n"
			text += v
		drawCard(img, text, center, line[0][0:-4] + ".pdf")




	
