# given image rgb, depth, and bbox_list.json
# draw pcl and bbox on the image using open3d

import argparse
import open3d as o3d

import numpy as np
import cv2
import os
import json

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
offset_h=-820
offset_d=-1250

SCALE = 1


def getRotationMatrix(w0: np.ndarray, theta: float = None):
    """Calculates rotation matrix using Rodrigue's formula (norm of the w is assigned to theta)"""
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

def rotate_around_x(x,y,z, theta):
    R = getRotationMatrix(np.array([1, 0, 0]), theta)

    A=np.identity(4)
    A[:3, :3]=R

    point = np.stack((x, y, z)).reshape(1, 3)

    point_rotated = homogeneousMultiplication(A, point)
    
    x_rotated, y_rotated, z_rotated = point_rotated[0]

    return x_rotated, y_rotated, z_rotated

def convert_from_uvd(u, v, d, is_np=False):
    """
    u, v, d -> x, y, z

    +———— x(u)
    |
    |
    y(v)
    """
    # FALSE!!
    # d *= scale
    # x_over_z = (u - cx) / f
    # y_over_z = (v - cy) / f
    # z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    # x = x_over_z * z
    # y = y_over_z * z
    
    # z = d * scale
    # x = (u - cx) * z / f
    # y = (v - cy) * z / f

    d_= d * SCALE
    x = (u - CX) * d_ / FX
    y = - (v - CY) * d_ / FY
    z = - d_

    if is_np:
        return np.stack((x, y, z), axis=1)
    else:
        return x, y, z


def main(args):
    rgb = cv2.imread(args.rgb)
    depth = cv2.imread(args.depth, -1)
    counter_in_box = 0

    with open(args.bbox_json) as f:
        bbox_dict = json.load(f)

    bbox_array=np.zeros((len(bbox_dict), 6))
    for i, obj in enumerate(bbox_dict):
        bbox=obj["bounding_box"]
        bbox_array[i] = np.array([bbox["x0"], bbox["x1"], bbox["y0"], bbox["y1"], bbox["z0"], bbox["z1"]])
        print(obj["class_name"], bbox_array[i].round(2))


    np_pcl = np.zeros((depth.shape[0], depth.shape[1], 6))
    for i in range(depth.shape[0]): # y(v)
        for j in range(depth.shape[1]): # x(u)
            
            d = depth[i, j][2] + depth[i, j][1]*256 + depth[i, j][0]*256*256
            x,y,z = convert_from_uvd(j, i, d)

            # very inefficient -> use numpy and vectorize
            x, y, z = rotate_around_x(x, y, z, -angle * np.pi / 180)

            np_pcl[i, j, 0] = x
            np_pcl[i, j, 1] = y
            np_pcl[i, j, 2] = z
            
            # np_pcl[i, j, 3:] = [0, 100, 0]
            # np_pcl[i, j, 3:] = rgb[i, j, ::-1]
            # Soluk çıksın diye 
            np_pcl[i, j, 3] = max(0, rgb[i, j, 2] - 20)
            np_pcl[i, j, 4] = max(0, rgb[i, j, 1] - 20)
            np_pcl[i, j, 5] = max(0, rgb[i, j, 0] - 20)
                

            for bbox in bbox_array:
                if bbox[0] <= x <= bbox[1] and bbox[2] <= y <= bbox[3] and bbox[4] <= z <= bbox[5]:
                    counter_in_box += 1
                    if counter_in_box % 100 == 0:
                        print(".", end="", flush=True)
                    np_pcl[i, j, 3:] = [255, 0, 0]
                    np_pcl[i, j, 3] = min(255, np_pcl[i, j, 3] + 50)


    np_pcl = np_pcl.reshape(-1, 6)
    
    pcl = o3d.geometry.PointCloud()

    pcl.points = o3d.utility.Vector3dVector(np_pcl[:, :3])
    print("min x:", np_pcl[:, 0].min(), "max x:", np_pcl[:, 0].max())
    print("min y:", np_pcl[:, 1].min(), "max y:", np_pcl[:, 1].max())
    print("min z:", np_pcl[:, 2].min(), "max z:", np_pcl[:, 2].max())

    pcl.colors = o3d.utility.Vector3dVector(np_pcl[:, 3:]/255)
    
    things_to_draw = [pcl]
    # draw bbox
    for bbox in bbox_array:
        # xyz_center = np.array([(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2])
        # xyz_size = np.array([abs(bbox[1]-bbox[0]), abs(bbox[3]-bbox[2]), abs(bbox[5]-bbox[4])])
        p0=np.array([bbox[0], bbox[2], bbox[4]])
        p1=np.array([bbox[1], bbox[3], bbox[5]])

        bbox = o3d.geometry.AxisAlignedBoundingBox(p0, p1)
        bbox.color = (0, 0, 1)
        bbox
        things_to_draw.append(bbox)

    o3d.visualization.draw_geometries(things_to_draw)



if __name__ == "__main__":
    
    data= "subject_2/task_1_k_cooking/take_1"

    rgb_default=f"/home/enerd/bimanual/bimacs_rgbd_data/bimacs_rgbd_data/{data}/rgb/chunk_0/frame_0.png"

    depth_default=f"/home/enerd/bimanual/bimacs_rgbd_data/bimacs_rgbd_data/{data}/depth/chunk_0/frame_0.png"

    bbox_default=f"/home/enerd/bimanual/pgt_bimanual/raw/bimacs_derived_data/{data}/3d_objects/frame_0.json"

    parser = argparse.ArgumentParser(description="Draw pcl and bbox on the image using open3d")
    parser.add_argument("--rgb", help="Path to the rgb image", default=rgb_default)
    parser.add_argument("--depth", help="Path to the depth image", default=depth_default)
    parser.add_argument("--bbox_json", help="Path to the 3d bbox json", default=bbox_default)

    args = parser.parse_args()
    

    main(args)