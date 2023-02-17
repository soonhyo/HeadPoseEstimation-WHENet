import numpy as np
import cv2
from whenet import WHENet
from utils import draw_axis
import os
import argparse
from yolo_v3.yolo_postprocess import YOLO
from PIL import Image
from scipy.stats import norm

import open3d as o3d
import quaternion
import rospy

def process_detection( model, img, bbox, args ):

    y_min, x_min, y_max, x_max = bbox
    # enlarge the bbox to include more background margin
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, img.shape[1])

    img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)
    
    x_center = int((x_min + x_max )/2)/10000
    y_center = int((y_min + y_max )/2)/10000
    
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
    
    draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )
    
    if args.display == 'full':
        cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    return img, roll, pitch, yaw, x_center, y_center

def sigmoid(x, a):
    return  1/ (1+ np.exp(-a*x))

def hpf(input, th):

    return input if abs(input) > th else 0

def look_at_view(vis):
    ctr = vis.get_view_control()
    ctr.set_lookat(np.array([0, 0, 0]))
    return True

def main(args):
    whenet = WHENet(snapshot=args.snapshot)
    yolo = YOLO(**vars(args))
    VIDEO_SRC = 0 if args.video == '' else args.video # if video clip is passed, use web cam
    cap = cv2.VideoCapture(0)
    print('cap info',VIDEO_SRC)
    ret, frame = cap.read()
    print(cap.isOpened())

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output, fourcc, 30, (frame.shape[1], frame.shape[0]))  # write the result to a video

    head_mesh = o3d.io.read_triangle_mesh("head_model.stl")
    head_mesh.compute_vertex_normals()
    # R = head_mesh.get_rotation_matrix_from_axis_angle(np.array([-np.pi/2, 0, 0]))

    #head_mesh.rotate(R)
    
    head_pcd = head_mesh.sample_points_poisson_disk(number_of_points=500, init_factor=10)
    head_pcd.paint_uniform_color([1, 0, 0])

    vis = o3d.visualization.VisualizerWithEditing()
    
    #vis = o3d.visualization.Visualizer()
    vis.create_window()
   # vis.add_geometry(head_mesh)
    vis.add_geometry(head_pcd)
           
    b_r = 0
    b_p = 0
    b_y = 0
    b_x = 0
    b_y = 0
    
    b_R = np.eye(3)

    th = 0.05 # rad
    while True:
        try:
            ret, frame = cap.read()
        except:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        bboxes, scores, classes = yolo.detect(img_pil)
        for bbox in bboxes:
            frame, r, p, y, x_c, y_c = process_detection(whenet, frame, bbox, args)
        cv2.imshow('output',frame)
        out.write(frame)

        r =np.deg2rad(r)
        p =np.deg2rad(p) 
        y =np.deg2rad(y)

        r = b_r + hpf(r - b_r, th)
        p = b_p + hpf(p - b_p, th)
        y = b_y + hpf(y - b_y, th)
        x_c = b_x + hpf(x_c - b_x, th)
        y_c = b_y + hpf(y_c - b_y, th)
         
        print(r, p, y)

        #for mesh
        # R = head_mesh.get_rotation_matrix_from_axis_angle(np.array([-p,r,-y]))
        # head_mesh.rotate(b_R.T)
        # head_mesh.rotate(R)

        # head_mesh.translate(np.array([-b_x, -b_y, 0]))
        # head_mesh.translate(np.array([x_c, y_c, 0]))

        #for pcd
        R = head_pcd.get_rotation_matrix_from_axis_angle(np.array([-p,r,y]))
        head_pcd.rotate(b_R.T)
        head_pcd.rotate(R)

        head_pcd.translate(np.array([-b_x, -b_y, 0]))
        head_pcd.translate(np.array([x_c, y_c, 0]))

        vis.update_geometry(head_mesh)

        # vis.update_geometry(head_pcd)

        vis.poll_events()
        vis.update_renderer()
        
        b_R = R
        b_r = r
        b_p = p
        b_y = y
        b_x = x_c
        b_y = y_c
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='whenet demo with yolo')
    parser.add_argument('--video', type=str, default='IMG_0176.mp4', help='path to video file. use camera if no file is given')
    parser.add_argument('--snapshot', type=str, default='WHENet.h5', help='whenet snapshot path')
    parser.add_argument('--display', type=str, default='simple', help='display all euler angle (simple, full)')
    parser.add_argument('--score', type=float, default=0.3, help='yolo confidence score threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='yolo iou threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--output', type=str, default='test.avi', help='output video name')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
