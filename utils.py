import matplotlib.pyplot as plt


def save_img(img, step, octave):
  """Save image"""
  plt.figure()
  plt.imshow(img)
  plt.axis('off')
  plt.title('Step {} Octave {}'.format(step, octave))
  plt.savefig('./sample/{}_step_{}_octave.jpg'.format(step, octave))
  plt.close('all')
