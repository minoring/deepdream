import matplotlib.pyplot as plt


def imshow(img):
  """Display an image"""
  plt.figure()
  plt.grid(False)
  plt.axis('off')
  plt.imshow(img)
  plt.show()
