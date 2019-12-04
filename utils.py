import glob

import imageio
import matplotlib.pyplot as plt


def save_img(img, title):
  """Save image"""
  plt.figure()
  plt.imshow(img)
  plt.axis('off')
  plt.title(title)
  plt.savefig('./sample/{}.jpg'.format(title))
  plt.close('all')


def create_gif():
  """Create gif using saved images"""
  anim_file = 'sample/training.gif'

  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('sample/*.jpg')
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
