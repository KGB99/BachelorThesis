import argparse
import os

def filterPowerDrillModal(x):
    powerdrill_id = "000000" # we are only interested in the powerdrill bitmask for now
    check = x.split("_")[1]
    check = check.split(".")[0]
    print('Modal BitMask ' + str(globals()["cur_id"]) + '/' + str(len_modalDirList))
    globals()["cur_id"] += 1
    if check == powerdrill_id:
        globals()["modalList"].append(x.split("_")[0])
        return True
    else:
        return False
    
def filterPowerDrillAmodal(x):
    powerdrill_id = "000000" # we are only interested in the powerdrill bitmask for now
    check = x.split("_")[1]
    check = check.split(".")[0]
    print('Amodal BitMask ' + str(globals()["cur_id"]) + '/' + str(len_amodalDirList))
    globals()["cur_id"] += 1
    if check == powerdrill_id:
        globals()["amodalList"].append(x.split("_")[0])
        return True
    else:
        return False
    
def filterImageDirList(x):
    print('Filtering image ' + str(globals()["cur_filterImage"]) + '/' + str(len_imageDirList))
    globals()["cur_filterImage"] += 1   
    if x.split(".png")[0] in globals()["bitList"]:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program creates a list of the filtered images for the powerdrill and saves it as a .txt file')
    parser.add_argument("--image_parent", help="This is the path to the parent folder of all the scenes containing training data", required=True, type=str)
    args = parser.parse_args()
    parent_path = args.image_parent

    #status print
    print("Working on directory: " + parent_path)

    #needed global variable for status updates
    cur_id = 1
    modalList = []
    amodalList = []

    parentDirList = os.listdir(parent_path)
    len_parentDirList = len(parentDirList)
    for cameraNr, camera in enumerate(parentDirList[:1]):
        bitmask_modal = parent_path + '/' + camera + '/mask'
        bitmask_amodal = parent_path + '/' + camera + '/mask_visib'
        image_path = parent_path + '/' + camera + '/rgb'

        #create a list of all bitmasks and filter the powerdrill images, 
        #then make sure only those images that have corresponding masks are included in training annotation
        print('Filtering the powerdrill in the bitmasks of ' + camera)
        print('Filtering camera ' + str(cameraNr + 1) + '/' + str(len_parentDirList))

        modalDirList = os.listdir(bitmask_modal)
        len_modalDirList = len(modalDirList)
        
        amodalDirList = os.listdir(bitmask_amodal)
        len_amodalDirList=  len(amodalDirList)
        
        #imageDirList = os.listdir(image_path)
        #len_imageDirList = len(imageDirList)

        #filter modal and amodal masks
        modalDirList = list(filter(filterPowerDrillModal, modalDirList))
        modalDirList = sorted(modalDirList)

        cur_id = 1

        amodalDirList = list(filter(filterPowerDrillAmodal, amodalDirList))
        amodalDirList = sorted(amodalDirList)

        cur_id = 1

        f = open('filtered_lists/filtered_bitmasks_' + camera + '.txt', 'w')
        f.write(str(len(modalDirList)) + '\n')
        for line in modalDirList:
            f.write(line + '\n')
        f.write(str(len(amodalDirList)) + '\n')
        for line in amodalDirList:
            f.write(line + '\n')

        f.close()






