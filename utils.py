import cv2 as cv
import numpy as np

def indicate_points(path, filename):
    """
    Display image to indicate object edges.

    :param path: Filepath.
    :param filename: Filename.
    :return: Indicate points.
    """

    img = cv.imread(path)
    mouseX, mouseY = -1, -1
    file = open(filename, 'w')

    def draw_circle(event,x,y,flags,param):
        global mouseX,mouseY
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(img,(x,y), 1, (255,0,0), 3)
            mouseX,mouseY = x,y
            file.write(f'{mouseX}   {mouseY}\n')

    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_circle)

    while(1):
        cv.imshow('image', img)
        k = cv.waitKey(20) & 0xFF
        if k == 27:
            file.close()
            break

def load_points(filename):
    """
    Loading given object points.

    :param filename: Filename.
    :return: Array of given points.
    """
    
    points = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 0:
                pass
            else:
                line = list(map(int, line))
                points.append(line)
    points = np.asarray(points)
    return points

def direct_linear_calibration(object_points, img_points):
    """
    Create camera calibration matrix.

    :param object_points: Given object points.
    :param img_points: Obtained image points.
    :return: Calibration matrix.
    """
    X, Y, Z = object_points[:, 0], object_points[:, 1], object_points[:, 2]
    u, v = img_points[:, 0], img_points[:, 1]

    i = 0
    n = len(img_points)
    UDV = np.zeros((2*n, 12))
    for j in range(0, n):
        UDV[i, :] = [X[j], Y[j], Z[j], 1, 0, 0, 0, 0, -u[j]*X[j], -u[j]*Y[j], -u[j]*Z[j], -u[j]]
        UDV[i+1, :] = [0, 0, 0, 0, X[j], Y[j], Z[j], 1, -v[j]*X[j], -v[j]*Y[j], -v[j]*Z[j], -v[j]]
        i += 2
    return UDV

def projection_matrix(UDV):
    """
    Calculate the projection matrix based on the product of UDV using SVD.

    :param UDV: Direct linear calibration.
    :return: Translation matrix, rotation matrix and intrinsic projection.
    """

    _, _, vh = np.linalg.svd(UDV, full_matrices=False)
    M = vh[-1, :]
    M = np.reshape(M, (3, 4)) 
    return M

def get_camera_params(M):
    """
    Retrives Camera parameters from projection Matrix.

    :param M: Projection matrix.
    :param right_image: The right image.
    :return: Translation matrix, rotation matrix and intrinsic projection.
    """
    m1, m2, m3 = M[0, :3], M[1, :3], M[2, :3]
    
    cx = np.dot(m1, m3)
    cy = np.dot(m2, m3)
    fx = np.linalg.norm(np.cross(m1, m3))
    fy = np.linalg.norm(np.cross(m2, m3))
    C = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K = np.dot(C, np.hstack((np.eye(3), np.zeros((3, 1)))))

    r1 = 1 / fx * (m1 - cx * m3)
    r2 = 1 / fy * (m2 - cy * m3)
    r3 = m3
    R = np.array([[*r1], [*r2], [*r3]])

    tx = 1 / fx * (M[0, 3] - cx * M[2, 3])
    ty = 1 / fy * (M[1, 3] - cx * M[2, 3])
    tz = M[2, 3]
    T = np.array([[tx], [ty], [tz]])
    return [K, R, T]

class Camera:
    """
    The Camera class that owns the parameter, points and matrices.
    """

    def __init__(self, image_path, corresponding_points, object_points):
        """
        :param image_path: Path to image file.
        :param corresponding_points: Camera specific object points obtained from the image.
        :param object_points: Given points of the object.
        """
        self.path = image_path
        self.corresponding_points = corresponding_points
        self.obj_points = object_points
        self.B = direct_linear_calibration(self.obj_points, self.corresponding_points[:12])
        self.M = projection_matrix(self.B)
        self.K, self.R, self.T = get_camera_params(self.M)
        self.A = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1])) 
        self.M_ka = np.dot(self.K, self.A) 
        self.img = cv.imread(self.path)

def draw_axes(img, M, axis_len):
    """
    Visulizes the axes in the picture.

    :param img: The image.
    :param M: projection_matrix.
    :param axis_len: Length of the axis.
    :return: An image with axis.
    """

    og3D_h = np.array([*(0, 0, 0), 1])
    x3D_h = np.array([axis_len, 0, 0, 1])
    y3D_h = np.array([0, axis_len, 0, 1])
    zaxis3D_h = np.array([0, 0, axis_len, 1])

    og2D_h = M.dot(og3D_h.T)
    x2D_h = M.dot(x3D_h.T)
    y2D_h = M.dot(y3D_h.T)
    z2D_h = M.dot(zaxis3D_h.T)

    og2D = tuple(map( int, og2D_h[:2]/og2D_h[2]))
    x2D = tuple(map( int, x2D_h[:2]/x2D_h[2]))
    y2D = tuple(map( int, y2D_h[:2]/y2D_h[2]))
    z2D = tuple(map( int, z2D_h[:2]/z2D_h[2]))

    img = cv.line(img, og2D, x2D, color=(255, 0, 0), thickness=2)
    img = cv.line(img, og2D, y2D, color=(0, 255, 0), thickness=2)
    img = cv.line(img, og2D, z2D, color=(0, 0, 255), thickness=2)
    return img

def homogeneous_coordinates(left_image, right_image):
    """
    Calculates the homogeneous coordinates based on two given images.

    :param left_image: The left image.
    :param right_image: The right image.
    :return: the calculated points.
    """

    m1l, m2l, m3l = left_image.M_ka[0], left_image.M_ka[1], left_image.M_ka[2]
    m1r, m2r, m3r = right_image.M_ka[0], right_image.M_ka[1], right_image.M_ka[2]
    X_h = []

    for (left_points, right_points) in zip(left_image.corresponding_points, right_image.corresponding_points):
        xl, yl = left_points[0], left_points[1]
        xr, yr = right_points[0], right_points[1]
        P = np.zeros((4, 4))
        P[0, :] = xl*m3l - m1l
        P[1, :] = yl*m3l - m2l
        P[2, :]  = xr*m3r - m1r
        P[3, :]  = yr*m3r - m2r
        _, _, vh = np.linalg.svd(P)
        solution = vh[-1, :]
        X_h.append(solution)
    X_h = np.asarray(X_h)
    estimated_points = X_h[:, :3]/np.expand_dims(X_h[:, 3], axis=1)
    return estimated_points