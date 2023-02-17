import numpy as np


def frame2_make_mask(x, y, r, trans, sz):  # sz is height by width
    x, y, r, trans = float(x), float(y), float(r), float(trans)
    if r > 1:
        return np.ones(sz)
    else:
        radius = r * sz[1]
        transition = trans * sz[1]
        x_, y_ = x, y
        x = np.linspace(-sz[1] / 2, sz[1] / 2, sz[1]) - x_ * sz[0]
        y = np.linspace(-sz[0] / 2, sz[0] / 2, sz[0]) - y_ * sz[0]
        [X, Y] = np.meshgrid(x, y)
        rr = np.sqrt(X * X + Y * Y)
        fxn = lambda r: 0.5 * (1 + np.cos(np.pi * r)) * (r < 1) * (r > 0) + (r < 0)
        alpha_mask = fxn((rr - radius) / transition + 1)
        return alpha_mask


def frame2_apply_mask(params, frame):
    frame_size = frame.T.shape  # frame is height by width
    alpha_mask = frame2_make_mask(
        params["aperture_x"],
        params["aperture_y"],
        params["aperture_r"],
        params["aperture_transition"],
        frame_size,
    )
    bg = params["background_value"]
    img = (frame - bg) * alpha_mask.T + bg
    return img
