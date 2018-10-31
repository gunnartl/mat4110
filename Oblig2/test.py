import imageio as im
import numpy as np

im1 = np.array(im.imread("board.png",as_gray= True),float)/255
im2 = np.array(im.imread("manet.jpg", as_gray = True),float)/255
im3 = np.array(im.imread("njåk.jpg", as_gray = True),float)/255

focus = im2

u,s,vh = np.linalg.svd(focus, full_matrices = False)
u,s1,vh = np.linalg.svd(im1, full_matrices = False)
u,s2,vh = np.linalg.svd(im2, full_matrices = False)

r = 40

compressed = np.dot(u[:,:r]*s[:r],vh[:r,:])

rate = focus.size/(u[:,:r].size+s[:r].size+vh[:r,:].size)

print("r = %i,"%r,"comression ratio = %.2f"%rate)

print(s)

import matplotlib.pyplot as plt
plt.plot(np.log10(s))
plt.plot(np.log10(s1))
plt.plot(np.log10(s2))
plt.title("Singular values for New York")
plt.ylabel("Log$_{10}$(size)")
plt.show()

plt.subplot(2,1,1)
plt.title("Original image")
plt.imshow(focus)
plt.axis("off")

plt.subplot(2,1,2)
plt.title("Compressed image with r = %i"%r)
plt.imshow(compressed)
plt.axis("off")
plt.show()