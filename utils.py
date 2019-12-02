import matplotlib.pyplot as plt


def save_img(img, step):
  """Save an image"""
  plt.figure()
  plt.grid(False)
  plt.axis('off')
  plt.imshow(img)
  plt.savefig('./sample/{}step_img.jpg'.format(step))
  plt.clf()
