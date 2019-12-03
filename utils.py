import imageio
import glob

import matplotlib.pyplot as plt


def save_img(img, step, octave):
  """Save image"""
  plt.figure()
  plt.imshow(img)
  plt.axis('off')
  plt.title('Step {} Octave {}'.format(step, octave))
  plt.savefig('./sample/{}_step_{}_octave.jpg'.format(step, octave))
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
  