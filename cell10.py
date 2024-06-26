probabilities = model.predict(test_data_gen)

def plotImages(images_arr, probabilities=None):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        if probabilities is not None:
            title = '%.2f%%' % (probabilities[img_idx] * 100)
            ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

test_images, _ = next(test_data_gen)
plotImages(test_images[:5], probabilities[:5])
