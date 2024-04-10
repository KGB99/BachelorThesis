# BachelorThesis

This code is currently in the process of being cleaned up and made presentable. The code has been used for the bachelor thesis of Kerim Birgi under supervision of Jonas Hein and Prof. Dr. Marc Pollefeys in the Computer Vision and Geometry Lab at ETH Zurich

## Dataset storage in Harddrive
- Obj\_id is as follows
	- powerdrill = 1
	- screwdriver = 2
	- hololens\_1 = 3
	- spine? = 4
	- spine2? = 5
	- spine3? = 6
    
## labels created from makeCoco have following format:
- labels.json
    - cameras
        - images (id)
            - gt_exists
                - int of value 0 or 1
            - img
                - id
                - width
                - height
                - file_name
            - mask
                - segmentation
                - bbox
                - area
                - iscrowd
                - image_id
                - category_id
                - id


