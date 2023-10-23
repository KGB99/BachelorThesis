# BachelorThesis

## Dataset storage in Harddrive
- The dataset for training is in mvpsp/train
- The train set has multiple different scenes each with a unique id and a directory
- Inside each scenes directory there is multiple sub directories
	- depth has depth photos
	- mask has mask photos of the entire tools including the occluded parts in the scene, furthermore there are multiple masks so it goes xxx\_yyy where xxx is the frame nr. and yyy is the object nr.
		- 000000 is the power drill
		- 000001 is a hololens
		- 000002 is the other hololens 
		- 000003 is the spine  
	- mask\_visib has masks of only the visible parts of the instruments
	- rgb has the normal rgb images which are needed for eval and train
- we train with rgb and masks and then validate by testing on rgbs and measuring the IOU of the calculated masks and actual masks from the test set

TODO:
- Change makeCoco so that it just does one iteration through imageListDir and skips images for training that do not have the bitmask for the drill
- Play around with cpu and gpu amount requests to speed up training, try 2 gpus next time and more cpu computation space.
