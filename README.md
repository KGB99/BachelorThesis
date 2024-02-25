# BachelorThesis

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
                - filename
            - mask
                - segmentation
                - bbox
                - area
                - iscrowd
                - image_id
                - category_id
                - id


