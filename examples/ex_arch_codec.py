import caterva
import numpy as np
from time import time

from PIL import Image
import matplotlib.pyplot as plt


image = Image.open("goldhill1.tif")
image = image.resize((600, 600)).convert("F")

data = np.asarray(image).astype(np.float64)

f, axs = plt.subplots(3, 3, figsize=(11, 12))
axs = axs.flatten()

axs[0].imshow(data, cmap="gray")
axs[0].title.set_text(f"Original")
axs[0].axis("off")

blocks = (100, 100)
chunks = (200, 200)

for i, ax in enumerate(axs[1:]):
    print(i + 2)

    t0 = time()
    c = caterva.asarray(data,
                        chunks=chunks,
                        blocks=blocks,
                        codec=caterva.Codec.AACODEC,
                        codecmeta=i + 2,
                        filters=[],
                        filtersmeta=[],
                        nthreads=1,
                        )

    t1 = time()
    c_time = t1 - t0

    ratio = c.cratio

    t0 = time()
    t_data = c[:].view(data.dtype)
    t1 = time()
    d_time = t1 - t0

    ax.imshow(t_data, cmap="gray")

    # Draw chunks and blocks
    for l in range(blocks[1], data.shape[1], blocks[1]):
        ax.vlines(x=l, ymin=0, ymax=data.shape[0], colors="red", linewidth=1)
    for l in range(blocks[0], data.shape[0], blocks[0]):
        ax.hlines(y=l, xmin=0, xmax=data.shape[1], colors="red", linewidth=1)

    for l in range(chunks[1], data.shape[1], chunks[1]):
        ax.vlines(x=l, ymin=0, ymax=data.shape[0], colors="blue", linewidth=2)
    for l in range(chunks[0], data.shape[0], chunks[0]):
        ax.hlines(y=l, xmin=0, xmax=data.shape[1], colors="blue", linewidth=2)


    ax.title.set_text(f"{i + 2} archetypes ({ratio:.2f}x)\n"
                      f"- comp: {c_time:.2f} s\n"
                      f"- decomp: {d_time:.6f} s")
    ax.axis("off")

plt.show()
