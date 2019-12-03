import matplotlib.pyplot as plt


def save_img(img, step, loss):
  """Save image"""
  plt.figure()
  plt.imshow(img)
  plt.axis('off')
  plt.title('Step {} Loss {}'.format(step, loss))
  plt.savefig('./sample/{}step_img.jpg'.format(step))
  plt.close('all')
