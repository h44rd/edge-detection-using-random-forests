import cv2
import numpy as np
import scipy

def rgb2luv(src):
    assert src.dtype == np.float64 or src.dtype == np.float32
    assert src.ndim == 3 and src.shape[-1] == 3

    a = 29.0 ** 3 / 27
    y0 = 8.0 / a
    maxi = 1.0 / 270

    table = [i / 1024.0 for i in xrange(1025)]
    table = [116 * y ** (1.0 / 3.0) - 16 if y > y0 else y * a for y in table]
    table = [l * maxi for l in table]
    table += [table[-1]] * 39

    rgb2xyz_mat = np.asarray([[0.430574, 0.222015, 0.020183],
                             [0.341550, 0.706655, 0.129553],
                             [0.178325, 0.071330, 0.939180]])
    xyz = np.dot(src, rgb2xyz_mat)
    nz = 1.0 / (xyz[:, :, 0] + 15 * xyz[:, :, 1] + 3 * xyz[:, :, 2] + 1e-35)

    L = [table[int(1024 * item)] for item in xyz[:, :, 1].flatten()]
    L = np.asarray(L).reshape(xyz.shape[:2])
    u = L * (13 * 4 * xyz[:, :, 0] * nz - 13 * 0.197833) + 88 * maxi
    v = L * (13 * 9 * xyz[:, :, 1] * nz - 13 * 0.468331) + 134 * maxi

    luv = np.concatenate((L[:, :, None], u[:, :, None], v[:, :, None]), axis=2)
    return luv.astype(src.dtype, copy=False)

def gradient(src, norm_radius=0, norm_const=0.01):
    if src.ndim == 2:
        src = src[:, :, None]

    dx = np.zeros(src.shape, dtype=src.dtype)
    dy = np.zeros(src.shape, dtype=src.dtype)
    for i in xrange(src.shape[2]):
        dy[:, :, i], dx[:, :, i] = np.gradient(src[:, :, i])

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    idx_2 = np.argmax(magnitude, axis=2)
    idx_0, idx_1 = np.indices(magnitude.shape[:2])
    magnitude = magnitude[idx_0, idx_1, idx_2]
    if norm_radius != 0:
        magnitude /= conv_tri(magnitude, norm_radius) + norm_const
    magnitude = magnitude.astype(src.dtype, copy=False)

    dx = dx[idx_0, idx_1, idx_2]
    dy = dy[idx_0, idx_1, idx_2]
    orientation = np.arctan2(dy, dx)
    orientation[orientation < 0] += np.pi
    orientation[np.abs(dx) + np.abs(dy) < 1e-5] = 0.5 * np.pi
    orientation = orientation.astype(src.dtype, copy=False)
    orientation = np.asarray(orientation)
    orientation = np.repeat(orientation[:,:,np.newaxis], 4, axis=2)
    return magnitude, orientation

def triangular_blur(src, radius):
    if radius == 0:
        return src
    elif radius <= 1:
        p = 12.0 / radius / (radius + 2) - 2
        kernel = np.asarray([1, p, 1], dtype=np.float64) / (p + 2)
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel,
                               borderType=cv2.BORDER_REFLECT)
    else:
        radius = int(radius)
        kernel = range(1, radius + 1) + [radius + 1] + range(radius, 0, -1)
        kernel = np.asarray(kernel, dtype=np.float64) / (radius + 1) ** 2
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel,
                               borderType=cv2.BORDER_REFLECT)

def blur(img):
    blur = cv2.blur(img,(5,5))
    return cv2.resize(blur, (0,0), fx=0.5, fy=0.5)

def compute_pairwise(src):
    out = np.zeros((300, 13))
    for ch in range(13):
        ind = 0
        for x in range(25):
            for y in range(x+1, 25):
                out[ind] = src[x/5][x%5][ch] - src[y/5][y%5][ch]
                ind += 1
    return out

def get_features_from_patch(patch):
    final = np.zeros((32, 32, 13))
    final[:, :, 0:3] = patch[:, :, :]
    # print patch.shape
    magnitude1, orientation1 = gradient(patch)
    magnitude2, orientation2 = gradient(cv2.resize(patch, (0,0), fx=2, fy=2))
    final[:, :, 3:4] = np.asarray(magnitude1)[:, :, np.newaxis]
    final[:, :, 4:5] = np.asarray(cv2.resize(magnitude2, (0,0), fx=0.5, fy=0.5))[:, :, np.newaxis]
    final[:, :, 5:9] = np.asarray(orientation1)
    final[:, :, 9:13] = np.asarray(cv2.resize(orientation2, (0,0), fx=0.5, fy=0.5))
    final1 = blur(final)
    final2 = triangular_blur(final, 8)
    final2 = compute_pairwise(final)
    final1 = final1.flatten()
    final2 = final2.flatten()
    out = np.append(final1, final2)
    return out

def test_preprocessing1(patch):
    img_size = np.array(patch).shape
    final = np.zeros((img_size[0], img_size[1], 13))
    final[:, :, 0:3] = patch[:, :, :]
    # print patch.shape
    magnitude1, orientation1 = gradient(patch)
    magnitude2, orientation2 = gradient(cv2.resize(patch, (0,0), fx=2, fy=2))
    final[:, :, 3:4] = np.asarray(magnitude1)[:, :, np.newaxis]
    final[:, :, 4:5] = np.asarray(cv2.resize(magnitude2, (0,0), fx=0.5, fy=0.5))[:, :, np.newaxis]
    final[:, :, 5:9] = np.asarray(orientation1)
    final[:, :, 9:13] = np.asarray(cv2.resize(orientation2, (0,0), fx=0.5, fy=0.5))
    return final

def test_preprocessing2(final):
    final1 = blur(final)
    final2 = triangular_blur(final, 8)
    final2 = compute_pairwise(final)
    final1 = final1.flatten()
    final2 = final2.flatten()
    out = np.append(final1, final2)
    return out

def get_lbls(patch):
    new_patch = np.array([[0 for i in range(32)] for i in range(32)])
    for x in range(32):
        for y in range(32):
            new_patch[x][y] = max(patch[0][x][y], patch[1][x][y], patch[2][x][y], patch[3][x][y])
    new_patch = scipy.misc.imresize(new_patch, 0.5)
    return new_patch

def get_train_data(train_data):
    final_x = []
    final_y = []
    for i, (img, bnds, segs) in enumerate(train_data):
        img = rgb2luv(img)
        img = np.asarray(img)
        for x in range(0, len(img)-32, 32):
            for y in range(0, len(img[0])-32, 32):
                final_x.append(get_features_from_patch(img[x:x+32, y:y+32]))

        bnds = np.asarray(bnds)
        for x in range(0, bnds.shape[1]-32, 32):
            for y in range(0, bnds.shape[2]-32, 32):
                final_y.append(get_lbls(bnds[:, x:x+32, y:y+32]))
    return final_x, final_y

def get_test_data(img):
    final_x = []
    img = rgb2luv(img)
    img = np.asarray(img)
    for x in range(0, len(img)-32, 32):
        for y in range(0, len(img[0])-32, 32):
            final_x.append(get_features_from_patch(img[x:x+32, y:y+32]))

    return final_x

