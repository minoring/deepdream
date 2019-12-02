import matplotlib.pyplot as plt


def save_img(images, step):
  """Save images"""
  fig, axarr = plt.subplots(1, len(images))
  for i in range(len(images)):
    axarr[i].imshow(images[i])
    axarr[i].axis('off')
    axarr[i].set_title('Layer {}'.format(i + 1))
  fig.savefig('./sample/{}step_img.jpg'.format(step))
  plt.close(fig)
