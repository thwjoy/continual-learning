import torchvision
import matplotlib.pyplot as plt

def display_batch(batch, nrows):
    grid_img = torchvision.utils.make_grid(batch.cpu(), nrow=nrows)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()