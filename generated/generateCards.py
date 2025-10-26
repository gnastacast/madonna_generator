import glob
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import cv2

cardWidth = 2.5
cardHeight = 4
margin = 0.3

width = cardWidth - margin * 2
height = cardHeight - margin * 2

imageRounding = margin
dpi = 170

def drawCard(img, text, origin, name):

    def label(xy, text):
        y = xy[1]  # shift y-value for label so that it's below the artist
        ax.text(xy[0],
                y,
                text,
                ha="center",
                va="bottom",
                family='monospace',
                style='italic',
                size=4)
    fig = plt.figure(figsize=(cardWidth, cardHeight),dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    fig.subplots_adjust(
        left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
    )
    ax.set_axis_off()
    # Add image
    ax.imshow(
        img,
        extent= np.hstack([
            origin + [-width/2, width/2],
            origin + [height/2 - width, height/2],
        ]),
        interpolation=None,
    )
    # round image corners
    fancybox = mpatches.FancyBboxPatch(
        origin - [width/2 - imageRounding/2, width - height/2 - imageRounding/2],
        width - imageRounding,
        width - imageRounding,
        boxstyle=mpatches.BoxStyle("Round", pad=(margin + imageRounding) / 2),
        edgecolor=(1,1,1),
        linewidth= margin / 2 * dpi,
        facecolor=(1,1,1,0),
        linestyle="solid",
    )
    ax.add_patch(fancybox)

    # Add text
    label(origin + [0, -height/2], text)

    # Draw card dimension
    fancybox = mpatches.FancyBboxPatch(
        origin - [cardWidth/2, cardHeight/2],
        cardWidth,
        cardHeight,
        boxstyle=mpatches.BoxStyle("Round", pad=0),
        edgecolor=(0,0,0),
        linewidth= 1,
        facecolor=(1,1,1,0),
        linestyle="solid",
    )
    ax.add_patch(fancybox)

    # add a dotted line
    fancybox = mpatches.FancyBboxPatch(
        origin - [width/2, height/2], width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=margin),
        edgecolor=(0,0,0),
        facecolor=(1,1,1,0),
        linewidth = 0.5,
        linestyle="dotted",
    )
    ax.add_patch(fancybox)
    # add a fancy box
    fancybox = mpatches.FancyBboxPatch(
        origin - [width/2, height/2], width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=margin - margin / 4),
        edgecolor=(.8,.8,.8),
        facecolor=(1,1,1,0),
        linestyle="solid",
    )
    ax.add_patch(fancybox)


    ax.set_xlim([-cardWidth/2, cardWidth/2])
    ax.set_ylim([-cardHeight/2, cardHeight/2])
    # plt.axis('equal')
    # plt.axis('off')
    # plt.tight_layout()
    plt.savefig(name, format="pdf")
    plt.close()

    # plt.show()
    # quit()

with open("mary_metadata.csv", 'r') as f:
    for i, line in enumerate(f.readlines()):
        line = line.split(',')
        origin = np.array([0,0])
        img = cv2.imread(line[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = ""
        for j, v in enumerate(line[1:]):
            if j % 5 == 0 and j > 0:
                text += "\n"
            v = v.strip()
            while(len(v) < 8):
                v = " " + v
            text += v
        drawCard(img, text, origin, line[0][0:-4] + ".pdf")
