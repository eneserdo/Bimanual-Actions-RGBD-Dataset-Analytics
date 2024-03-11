
import json
import numpy as np
import cv2

# From bimanual's code:
IMG_WIDTH=640
IMG_HEIGHT=480
FOV_X=54 / 180 * np.pi 
FOV_Y=45 / 180 * np.pi 

FX= IMG_WIDTH / (2 * np.tan(FOV_X/2)) # 628.035
FY= IMG_HEIGHT / (2 * np.tan(FOV_Y/2)) # 579.411
CX=IMG_WIDTH/2
CY=IMG_HEIGHT/2

angle=20
offset_rl=25
offset_h=-694
offset_d=-1300

SCALE = 1

COLORS =[
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),


    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    
    (200, 100, 0),
    (200, 0, 100),    
    (100, 0, 200),
    (100, 200, 0),
    (0, 100, 200),
    (0, 200, 100),
    
    (0, 0, 100),
    (0, 100, 0),
    (100, 0, 0),
]



def getRotationMatrix(w0: np.ndarray, theta: float = None):
    """Calculates rotation matrix using Rodrigue's formula (norm of the w is assigned to theta), (theta in radian)"""
    # FIXME modifies the w implicitly

    if theta is None:
        norm = np.linalg.norm(w0)
        w0 = w0 / norm
        theta = norm
    else:
        assert np.allclose(np.linalg.norm(w0), 1.0)

    W_hat = np.array([[0, -w0[2], w0[1]], [w0[2], 0, -w0[0]], [-w0[1], w0[0], 0]])

    # Rodrigue's formula
    R = (np.identity(3) + np.sin(theta) * W_hat + (1 - np.cos(theta)) * np.matmul(W_hat, W_hat))

    # sanity check
    assert np.allclose(np.linalg.det(R), 1.0)

    return R


# This function is basically used everywhere
def homogeneousMultiplication(A, x):
    """A.shape = [D, D], x.shape = [N, D-1];
    Calculates x' = A @ [x.T, 1].T
    then returns x' in inhomogeneous vector form
    """
    assert A.shape[0] == A.shape[1] and x.shape[1] == (A.shape[0] - 1)

    x = np.concatenate((x, np.ones((x.shape[0], 1))), 1)
    x = A @ x.T
    x = x[:-1] / x[-1]

    return x.T


def convert_from_xyz(x, y, z, is_np=False):
    """
    x, y, z -> u, v, d
    """

    if is_np:
        assert np.isinf(x).any() == False and np.isinf(y).any() == False and np.isinf(z).any() == False
        assert np.isnan(x).any() == False and np.isnan(y).any() == False and np.isnan(z).any() == False
    else:
        raise NotImplementedError("what are you doing?")    

    # rotate x,y,z point around x-axis 19 degree
    R=getRotationMatrix(np.array([1, 0, 0]), +np.pi*angle/180)
    A=np.identity(4)
    A[:3, :3]=R

    xyz=np.stack([x, y, z]).T

    rotated_xyz=homogeneousMultiplication(A, xyz)


    x, y, z = np.hsplit(rotated_xyz, 3)
    
    x, y, z = x.flatten(), y.flatten(), z.flatten()

    u = -FX * x / z + CX
    v = FY * y / z + CY

    mask_to_nan = (u < 0) | (u > IMG_WIDTH) | (v < 0) | (v > IMG_HEIGHT)     
    # print("u:", u)
    # print("v:", v)

    u[mask_to_nan] = np.nan
    v[mask_to_nan] = np.nan

    if is_np:
        return np.stack((u, v), axis=1)
    else:
        return u, v



def get_bboxed_image(image, bbox_list):
    # bbox_list: N x 6 : [x0, x1, y0, y1, z0, z1]
    # bbox_list -> vertices_xyz -> vertices_uv -> edges
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0

          
          +y   -z
          |   /
          |  /
          | /   
          |/
          .----> +x 
    """

    vertex_pairs = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    for b_idx, bbox in enumerate(bbox_list):
        if np.allclose(bbox, 0):
            continue

        vertices_xyz = np.array([[bbox[0], bbox[2], bbox[4]], 
                                 [bbox[0], bbox[2], bbox[5]], 
                                 [bbox[0], bbox[3], bbox[4]], 
                                 [bbox[0], bbox[3], bbox[5]], 
                                 [bbox[1], bbox[2], bbox[4]], 
                                 [bbox[1], bbox[2], bbox[5]], 
                                 [bbox[1], bbox[3], bbox[4]], 
                                 [bbox[1], bbox[3], bbox[5]]])
                              
        vertices_uv = convert_from_xyz(vertices_xyz[:, 0], vertices_xyz[:, 1], vertices_xyz[:, 2], is_np=True)
        
        # name each box
        cv2.putText(image, f"{b_idx}", (int(vertices_uv[6][0]), int(vertices_uv[6][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # get edges from vertices
        for v1, v2 in vertex_pairs:
            # draw edge from vertex_pairs[v1] to vertex_pairs[v2]

            if np.isnan(vertices_uv[v1]).any() or np.isnan(vertices_uv[v2]).any():
                print("Nan in bbox vertices!")
                continue

            p1=vertices_uv[v1].astype(int)
            # p1=np.flip(p1)

            p2=vertices_uv[v2].astype(int)
            # p2=np.flip(p2)

            try:
                cv2.line(image, p1, p2, COLORS[b_idx], 1)
            except:
                print("v1:", v1, "v2:", v2)
                print("p1:", p1, "p2:", p2)
                print("p1_xyz:", vertices_xyz[v1], "p2_xyz:", vertices_xyz[v2])
                raise ValueError("Error in cv2.line")


if __name__ == "__main__":
    BASE_DIR="/home/enerd/bimanual"

    # RGB_DIR=BASE_DIR + "/bimacs_rgbd_data/subject_2/task_1_k_cooking/take_1/rgb/chunk_0/frame_0.png"
    RGB_DIR="/home/enerd/bimanual/bimacs_rgbd_data/bimacs_rgbd_data/subject_2/task_1_k_cooking/take_1/rgb/chunk_0/frame_0.png"

    BB_DIR=BASE_DIR + "/pgt_bimanual/raw/bimacs_derived_data/subject_2/task_1_k_cooking/take_1/3d_objects/frame_0.json"

    rgb=cv2.imread(RGB_DIR)

    with open(BB_DIR) as f:
        objects_3d = json.load(f)

    # objects_3d=objects_3d[:1]

    bbox_array=np.zeros((len(objects_3d), 6))

    for i, obj in enumerate(objects_3d):
        bbox=obj["bounding_box"]
        bbox_array[i] = np.array([bbox["x0"], bbox["x1"], bbox["y0"], bbox["y1"], bbox["z0"], bbox["z1"]])
        # print(obj["class_name"])

    get_bboxed_image(rgb, bbox_array)

    # cv2.imwrite("bboxed.png", rgb)
    # print(rgb.shape)
    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()