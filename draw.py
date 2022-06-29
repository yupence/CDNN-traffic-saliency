
import matplotlib.pyplot as plt
import data_load


def draw_picture(picturePath, mask, savePath):
    plt.rcParams['figure.figsize'] = (12.8, 7.2)
    plt.axis('off')
    bottom = plt.imread(picturePath)
    plt.imshow(bottom)
    print(mask.shape)
    im1 = plt.imshow(bottom
                     # , cmap=plt.cm.gray
                     , interpolation='nearest'
                     # , extent=extent
                     )

    im2 = plt.imshow(mask
                     # ,cmap=plt.cm.viridis
                     , alpha=.5
                     , interpolation='bilinear'
                     # ,extent=extent
                     )
    plt.savefig(savePath, bbox_inches='tight', pad_inches=-0.1)
    plt.show()


if __name__ == "__main__":
    picture = r'F:/baidudownload/baidudownload/out1/002031.jpg'
    mask = data_load.getLabel(1,2031)
    save_path = r"./mask_photo/6.jpg"
    draw_picture(picture,mask,save_path)
