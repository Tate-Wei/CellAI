# Phase images hold float values that denote the phase shift for each pixel.
# In the cellface project we found that depending on how these values get
# mapped to RGB-colors, the percepted appearance can change quite a bit.
# For example, just mapping them to grayscale makes details hard to see. Also,
# it can be quite confusing when a shift of 1rad is plotted as one color in one
# image, and as another color in another image.
#
# Therefore, we worked on a colormap that works quite well for phase images
# and uses color tones close to traditional staining. You are of course free
# to develop your own colormap (or even use a different plotting framework
# than matplotlib), but you are also invited to reuse the one we created
# for the cellface project. The code is below, please make sure to use
# the colormap AND the norm to ensure uniform scaling of the images. More
# information about matplotlib colormaps can be found here:
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# They work through the cmap and norm parameters of imshow:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html


import numpy as np
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import colorsys

CellfaceStdNorm = clrs.Normalize(vmin=-4, vmax=14, clip=True)

CellfaceStdCMap = clrs.LinearSegmentedColormap.from_list(
    "CellFace Standard",
    [
        # Position               R     G     B     A
        (CellfaceStdNorm(-4.0), [0.65, 0.93, 1.00, 1.0]),  # Air bubbles
        (CellfaceStdNorm(0.0), [1.00, 0.97, 0.96, 1.0]),  # Background
    ] + [
        (
            CellfaceStdNorm(2 + p * (14 - 2)),
            [max(min(val, 1.0), 0.0) for val in list(
                colorsys.hsv_to_rgb(
                    (280 - 90 * p) / 360,  # Hue: From Pink to Purple
                    0.5 + 1 * p,  # Saturation: Pastel to fully saturated
                    (1 - p) ** 2,  # Value: From Bright to Black
                )
            )]
            + [1.0],
        )
        for p in np.linspace(0.0, 1.0, 20)
    ],
)

def rgba_to_rgb(image):
    alpha = image[:, :, 3]  # Extract the alpha channel
    rgb = image[:, :, :3]  # Extract the RGB channels

    # Calculate the RGB values with alpha blending
    background = np.ones_like(rgb) * 255  # White background
    blended = (1 - alpha[..., None]) * background + alpha[..., None] * rgb

    # Convert blended image to uint8
    blended_uint8 = blended.astype(np.uint8)

    return blended_uint8