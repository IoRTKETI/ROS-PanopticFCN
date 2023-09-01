#!/usr/bin/env python3

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog #HM

from predictor import VisualizationDemo####

from panopticfcn import add_panopticfcn_config # noqa

import rospy
import rospkg
import yaml
from cv_bridge import CvBridge

# ROS message
from sensor_msgs.msg import Image

from panopticFCN.msg import pano_msg as Panoptic


WINDOW_NAME = "COCO detections"

def setup_cfg(cfg_path, confidence_threshold):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_panopticfcn_config(cfg) #HM    


    # ---------------label_17---------------
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").thing_classes = ["person", "car", "truck", "huge truck"] 
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").set(thing_train_id2contiguous_id={ 0: 11, 1: 12, 2: 13, 3: 14 }) 
    
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").stuff_classes = ["unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic sign", "vegetation", "terrain", "sky", "person", "car", "truck", "huge truck", "gas storage", "hazard storage"] #HM_original
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").set(stuff_train_id2contiguous_id={ 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16 }) #HM_change_trainId
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").stuff_colors = [(255, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 70), (13, 208, 131), (170, 60, 10), (230, 180, 70)] 


    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").thing_classes = ["person", "car", "truck", "huge truck"]
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").set(thing_val_id2contiguous_id={0: 11, 1: 12, 2: 13, 3: 14 })

    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").stuff_classes = ["unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic sign", "vegetation", "terrain", "sky", "person", "car", "truck", "huge truck", "gas storage", "hazard storage"] #HM_original
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").set(stuff_val_id2contiguous_id={ 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16 }) #HM_change_trainId
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").stuff_colors = [(255, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 70), (13, 208, 131), (170, 60, 10), (230, 180, 70)] 



    cfg.merge_from_file(cfg_path)
    
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()

    return cfg


class PanopticFCNdetection:
    def __init__(self):
        rospy.init_node('PanopticFCN_node')

        rospack = rospkg.RosPack()
        path = rospack.get_path('panopticFCN')


        yaml_file = 'PanopticFCN-R50-400-3x-FAST_hm.yaml'

        # subscribe image topic
        #subscribing_image_topic = '/video_source/raw'
        subscribing_image_topic = '/camera/color/image_raw'
        

        # publish visualization pano image
        self.visualization_Flag = True

        # threshold
        conf_threshold = 0.5

        # cfg setting
        path_yaml_file = f"{path}/config/{yaml_file}"
        cfg = setup_cfg(path_yaml_file, conf_threshold)
        self.demo = VisualizationDemo(cfg)

        # pub topic
        self.pano_info_pub = rospy.Publisher("/robot_data/pano_info", Panoptic, queue_size=1)
        if self.visualization_Flag == True:
            self.img_pub = rospy.Publisher("/robot_data/pano_img", Image, queue_size=10)

        self.bridge = CvBridge()
        self.rgb_image = None

        print("\nconfig file : {}".format(yaml_file))
        print("weights file : {}".format(cfg.MODEL.WEIGHTS))
        print("\nReady to predict!\n")

        # when subscribe Image
        rospy.Subscriber(subscribing_image_topic, Image, self.classify)
        rospy.spin()


    def pub_pano_info_(self, resp, time_):
        pano_msg = Panoptic()
        pano_msg.header.stamp = time_

        # panoptic ==instance
        pano_resp = resp["panoptic_seg"]
        inst_resp = resp["instances"]

        # seg_map = resp["seg_map"]
        seg_map = pano_resp[0].cpu().detach().numpy().astype("uint8")
        # h, w = seg_map.shape

        if(len(resp["instances"].pred_boxes)>=1):
        # pack semantic information
            info  = pano_resp[1]
            boxes = inst_resp.pred_boxes.tensor.cpu().detach().numpy().tolist()
        else:
            info  = pano_resp[1]
            boxes =[]

        boxes = np.reshape(boxes, len(boxes)*4)
        obj_id = []
        sem_id = []
        scores = []
        area = []
        obj_category = []
        sem_category = []

        for i in range(len(info)):
            info_current = info[i]
            if info_current["isthing"]:
                scores.append(info_current["score"])
                obj_category.append(info_current["category_id"])
                obj_id.append(info_current["id"])  
            else:
                area.append(int(info_current["area"]))
                sem_category.append(info_current["category_id"])
                sem_id.append(info_current["id"])

        pano_msg.height = len(seg_map)
        pano_msg.width = len(seg_map[0])
        seg_map = seg_map.reshape(pano_msg.height*pano_msg.width)
        pano_msg.seg_map = seg_map
        pano_msg.obj_id = obj_id
        pano_msg.obj_category = obj_category
        pano_msg.obj_scores = scores
        # 2D bounding boxes [x1 y1 x2 y2 ...]
        pano_msg.obj_boxes = [int(i) for i in boxes]

        pano_msg.sem_id = sem_id
        pano_msg.sem_category = sem_category
        #area = [int(value) for value in area]
        pano_msg.sem_area = area

        self.pano_info_pub.publish(pano_msg)


    # call back
    def classify(self, image):
        #self.rgb_image
        img = self.bridge.imgmsg_to_cv2(image, "bgr8")

        time_stamp = rospy.Time.now()
        start = rospy.get_time()

        if self.visualization_Flag == True:
            predictions, visualized_output = self.demo.run_on_image(img)
            opencv_image = visualized_output.get_image()
            if False : # used for image write
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite( 'test_result/result_pano_'+str(start)+'_.jpg', opencv_image)
            ros_image = self.bridge.cv2_to_imgmsg(opencv_image, encoding="rgb8")
            self.img_pub.publish(ros_image)

        else :
            predictions = self.demo.run_predictor(img)
            print("Inference Time : {:.3f} sec".format(rospy.get_time()-start))

        self.pub_pano_info_(predictions, time_stamp)
        print("Total Time : {:.3f} sec\n".format(rospy.get_time()-start))


if __name__ == '__main__':
    try:
        detector = PanopticFCNdetection()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')