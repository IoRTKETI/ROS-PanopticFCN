# ROS-PanopticFCN

This is a ROS package developed for **PanopticFCN in ROS topic images**. It's based on PanopticFCN with detectron2 and you can also use custom models if you want. The available weights are **pth files**.

<div align="center">
    <a href="./">
        <img src="./scripts/test_result/1image_rect_color18_leftImg8bit.jpg" width="59%"/>
    </a>
</div>

**Subscribe topic (type):**

Image (sensor_msgs/Image)

**Publish topic (type):**

panoptic_info (panopticFCN/pano_msg)

panoptic visualization Image (sensor_msgs/Image)


## Preferences
We tested in
**ROS noetic**.

``` shell
# clone to the ros catkin workspace
git clone https://github.com/IoRTKETI/ROS-PanopticFCN.git

# catkin make
cd ~/{your ros workspace PATH} && catkin_make

```


If you want publishing only panoptic info (not publishing panoptic Image).
``` shell
# ./panopticFCN/scripts/pano_main.py
# line 86
self.visualization_Flag = False
```

If your subscribed topic is different.
``` shell
# ./panopticFCN/scripts/pano_main.py
# line 83
subscribing_image_topic = '{your subscribed topic}'
```




## test
``` shell
# use roslaunch
roslaunch panopticFCN pano_test.launch

or

# use rosrun
roscore

rosrun panopticFCN pano_main.py

rosrun panopticFCN image_pub.py
```



## Use custom data

**Add weight file :** ./yolov7/scripts/model/custom file.pth

**Add yaml file :** ./yolov7/config/custom file.yaml

**Change yaml file path in python file :**
``` shell
# ./panopticFCN/scripts/pano_main.py
# line 79
yaml_file = 'custom file.yaml'
```
