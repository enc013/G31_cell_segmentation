########################################################################################################################
#Importing Packages:
import cv2
import math
import time
import argparse
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from tqdm import tqdm
import nd2reader as nd2
from os.path import join
########################################################################################################################
#Functions:

def lum_match(images, lum=(70, 32)):
    # Writing a function to match all the luminosity of all images in your list of images:
    # The two parameters are images, a list of images or a single images.

    # Checking if input is a list; if not, throw a value error to tell user that
    if type(images) is not list:
        raise ValueError("Value must be a list")

    output = images.copy()
    numim = len(output)

    M = lum[0]
    S = lum[1]

    # Looping through all our images to find mean M and standard deviation S:
    for i in range(0, numim):
        if output[i].ndim == 3:
            if output[i].shape[2] == 3:
                # Converting to grayscale if an RGB image
                output[i] = cv2.cvtColor(output[i], cv2.COLOR_RGB2GRAY)

                # Converting to float
                output[i] = output[i].astype(float)

                # Luminance matching
                if np.std(output[i]) != 0:

                    output[i] = ((output[i] - np.mean(output[i])) / np.std(output[i]) * S + M)

                else:
                    output[i][:] = lum[0]

                output[i] = output[i].astype('uint8')


            else:
                raise ValueError("The image is not a RGB image and is a on a color scale with two channels!")
        elif output[i].ndim == 2:
            # Converting to float
            output[i] = output[i].astype(float)

            if np.max(output[i]) > 255:
                # normalizing images
                output[i] = 255 * ((output[i] - np.min(output[i])) / (np.max(output[i]) - np.min(output[i])))

                # Luminance matching
            if np.std(output[i]) != 0:

                output[i] = (((output[i] - np.mean(output[i])) / np.std(output[i])) * S + M)

                # If calculation is more than max (255) or less than min (0) for
                # a gray scale image, we correct them
                output[i][output[i] > 255] = 255
                output[i][output[i] < 0] = 0

            else:
                output[i][:] = lum[0]

            # Converting to unsigned integer 8
            output[i] = output[i].astype('uint8')

    return output


def normalizeZStacks(z_stacks):  # Functional programmiong technique to avoid data duplication in RAM
    # Creating a new variable to store normalized images using lum_match:
    norm_zstacks = []

    # luminance matching and storing it to another variable by iterating and applying function lum_match:
    for z_stack in z_stacks:
        temp_var = lum_match(z_stack)
        norm_zstacks.append(temp_var)

    return norm_zstacks


def applyMedianFilter(z_stacks):
    # Defining a variable to store median filtered images.
    med_normzs = []

    # iterating over each z-stack
    for z_stack in z_stacks:

        # Defining a temporary variable:
        temp_list = []

        # iterating over images in each z-stack
        for image in z_stack:
            # defining a variable & applying median filter to each image:
            temp_var = cv2.medianBlur(image, 5)

            # appending to temporary list:
            temp_list.append(temp_var)

        # appending to new variable list med_normzs
        med_normzs.append(temp_list)

    return med_normzs


def applyTresholding(z_stacks):
    # Defining a variable to store all thresholded images:
    th_med_normzs = []

    for z_stack in z_stacks:

        # Defining temporary list to store images
        temp_list = []

        for image in z_stack:
            # Defining temporary variable
            th_val, temp_var = val_th, img_th = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)

            # Appending threshold image to temporary list:
            temp_list.append(temp_var)

        # Appending to our main variable
        th_med_normzs.append(temp_list)

    return th_med_normzs


def circularity(contour, area):
    #Circularity test to find circular objects
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        circularity_score = 0
    else:
        circularity_score = 4 * area * math.pi / (perimeter ** 2)

    return circularity_score > 0.30


def solidity(contour, area):
    #solidity score
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        solidity_score = 0
    else:
        solidity_score = area / hull_area

    return solidity_score > 0.90


def elongation(contour):
    #elongation score
    m = cv2.moments(contour)
    a = m['mu20'] + m['mu02']
    b = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
    if (a + b ** 0.5) == 0:
        elongation_score = 0
    else:
        elongation_score = (a - b ** 0.5) / (a + b ** 0.5)

    return abs(elongation_score) > 0.6


class Mass:
    def __init__(self, mId, zstackInd):
        # @mId Massd id (int)
        #
        # @blobs dict opf Blob2D instances where the key
        # is the Blob2D centroid
        #
        # @zstackInd related z_stack index (int)

        self.mId = mId
        self.blobs = {}
        self.zstackInd = zstackInd
        self._isAlone = False

    def add(self, blob):
        # @blob instance of Blob2d
        self.blobs[(blob.imageInd, blob.centroid)] = blob

    def getBlobDict(self):
        return self.blobs

    def isSphere(self):
        for _, blob in self.blobs.items():
            if not blob.isCircular():
                return False
        return True

    def setAlone(self, value):
        # @value bool
        self._isAlone = value

    def isAlone(self):
        # If true this means there is no masse over or
        # under it in the z_stack
        return self._isAlone

    def getBiggestRadiusBlob(self):
        # @return Blob2D
        biggest = None
        for bTup, blob in self.blobs.items():
            if not biggest:
                biggest = blob
            else:
                if blob.radius > biggest.radius:
                    biggest = blob
        return biggest


class Blob2D:
    def __init__(self, imageInd, imageIncInd, centroid, contour, areaMembers, radius, area, zstackInd):
        # @imageInd Image index (int)
        #
        # @centroid tuple (x,y)
        #
        # @contour is a list of tuples. A tuple contain the (x,y)
        # coord of a point on the contour
        #
        # @areaMembers is a matrix representying the image where when
        # coordinates' value are 1 indicates that they are within
        # the contour area
        #
        # @radius distance of the farthest point on the contour
        # from the center (int)
        #
        # @area area within the contour
        #
        # @zstackInd related z_stack index (int)

        self.imageInd = imageInd
        self.imageIncInd = imageIncInd
        self.centroid = centroid
        self.contour = contour
        self.areaMembers = areaMembers
        self.radius = radius
        self.area = area
        self.zstackInd = zstackInd

    def calculation(self):

        #Circularity:
        perimeter = cv2.arcLength(self.contour, True)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4*self.area*math.pi/(perimeter**2)

        #Solidity:
        hull = cv2.convexHull(self.contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            solidity = 0
        else:
            solidity = self.area/hull_area

        #Elongation:
        m = cv2.moments(self.contour)
        a = m['mu20'] + m['mu02']
        b = 4*m['mu11']**2 + (m['mu20']-m['mu02'])**2

        if (a+b**0.5) == 0:
            elongation = 0
        else:
            elongation = (a-b**0.5)/(a+b**0.5)

        results = [self.area, circularity, solidity, elongation]

        return results
    

    def isCircular(self):
        # Returns Bool
        x = circularity(contour=self.contour, area=self.area)

        # Combine five parameters together
        # All tests must return 1 (int) for this
        # to be true
        if x:  # x*y*z
            return True
        else:
            return False

    def isSoliditaryElongation(self):
        y = solidity(contour=self.contour, area=self.area)
        w = elongation(self.contour)

        if y * w == 1:
            return True
        else:
            return False

    def isOnEdge(self):
        h_img = self.areaMembers.shape[0]
        w_img = self.areaMembers.shape[1]

        rect = cv2.boundingRect(self.contour)

        x, y, w, h = rect

        if (x + w) >= (w_img - 10):
            return True
        elif x <= 100:
            return True
        elif (y + h) >= (h_img - 10):
            return True
        elif y <= 10:
            return True
        else:
            return False

            # perimeter = cv2.arcLength(self.contour, True)
        # if perimeter == 0:
        #    return False
        # area = cv2.contourArea(self.contour)
        # circularity = 4 * math.pi*(area/(perimeter * perimeter))

        # if circularity > 0.5:
        #    return True
        # return False


class BlobImage:
    # Blobs per Images
    def __init__(self, incInd):
        # @blobs list of Blob2D objects
        self._blobs = []

    def add(self, blob):
        # @blob Blob2D
        self._blobs.append(blob)

    def getBlobs(self):
        return self._blobs


class BlobStack:
    # BlobsListPerImage per Stacks
    def __init__(self, zstackInd):
        # @zstackInd related z_stack index (int)

        self.zstackInd = zstackInd

        # list of BPerImages objects
        self._blobImages = []

        # list of the related masses objects
        self._masses = []

    def add(self, bImage):
        self._blobImages.append(bImage)

    def getBlobImages(self):
        return self._blobImages

    def addMass(self, mass):
        self._masses.append(mass)

    def getMasses(self):
        return self._masses

    def numOfMasses(self):
        return len(self._masses)


def getAreaMembers(image, contour):
    # Determines which point is within the area of the contour
    #
    # @image image to process (matrix)
    #
    # @contour list of point coordinates which are on the line
    # of the contour
    #
    # @return Matrix representing which coordinates are inside
    # (value=1) or outside (value=0) of the contour

    # the result will be a matrix in which the inside of the contour is
    # is 1:

    #     mask = np.zeros(image.shape, np.uint8)

    #     result = cv2.drawContours(mask, contour, -1, (255),1)

    # Creating mask to export
    mask = np.zeros(image.shape, np.uint8)

    img_h = image.shape[0]
    img_w = image.shape[1]

    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    offset = 5
    if (x - offset) < 0:
        x1 = x
    else:
        x1 = x - offset

    # second x point
    if (x + w + offset) > img_w:
        x2 = x + w
    else:
        x2 = x + w + offset

    # first y point in bounding box
    if (y - offset) < 0:
        y1 = y
    else:
        y1 = y - offset

    # Second y point:
    if (y + h + offset) > img_h:
        y2 = y + h
    else:
        y2 = y + h + offset

    for i in range(y1, y2 + 1):  # img_h):
        for j in range(x1, x2 + 1):  # img_w)

            bool_cond = cv2.pointPolygonTest(contour, (j, i), False)

            if bool_cond >= 0:
                mask[i, j] = 255

    return mask


def distance(xi, xii, yi, yii):
    sq1 = (xi - xii) * (xi - xii)
    sq2 = (yi - yii) * (yi - yii)
    return math.sqrt(sq1 + sq2)


def getFarthest(centroid, contour):
    farthest = 0
    for coor in contour:
        val = distance(centroid[0], centroid[1], coor[0][0], coor[0][1])
        if val > farthest:
            farthest = val
    return farthest


def hasCommonArea(area1, area2):
    img_bwa = cv2.bitwise_and(area1, area2)

    if np.sum(img_bwa) > 0:
        return True
    else:
        return False


def isWithinRange(blob1, blob2):
    distRange = min(blob1.radius, blob2.radius)  # blob1.radius + blob2.radius - 20
    dist = distance(xi=blob1.centroid[0],
                    xii=blob2.centroid[0],
                    yi=blob1.centroid[1],
                    yii=blob2.centroid[1])
    if dist <= distRange:
        return True
    return False


def isAMember(blob, masses):
    # @blob Blob2D instance
    #
    # @masses list of Mass instances

    if not masses:
        return False

    for m in masses:
        for k, b in m.getBlobDict().items():
            if hasCommonArea(
                    area1=blob.areaMembers,
                    area2=b.areaMembers) and isWithinRange(
                blob1=blob,
                blob2=b) and (abs(blob.imageInd - b.imageInd) == 1):
                m.add(blob)
                return True
    return False


def isAlone(masse, bImages):
    # below /above test
    blKeys = masse.getBlobDict().keys()
    for bTup, blob in masse.getBlobDict().items():
        # btup = (blob.imageInd, blob.centroid)
        for bImInd in range(0, len(bImages)):
            if bImInd == blob.imageInd:
                continue
            bImage = bImages[bImInd]
            for iBlb in bImage.getBlobs():
                # is it part of the same masse?
                partOf = False
                for k in blKeys:
                    if k[0] == iBlb.imageInd and k[1] == iBlb.centroid:
                        partOf = True
                        break
                if partOf:
                    # move to the next iBlb
                    continue
                # Are they overlapping?
                if hasCommonArea(area1=blob.areaMembers, area2=iBlb.areaMembers):
                    return False
    return True


def generateMacroFile(macroFolder, zInd, iInd, nd2FilePath, imagePath, x, y, z):
    # This function creates the resulting macro file to be used with the microscope
    #
    # @masseId int
    #
    # @nd2FilePath nd2File complete path on disk
    #
    # @imagePath imaghe complete path on disk
    #
    # macros
    macroPath = "{0}_z{1}_{2}.mac".format(join(macroFolder, 'Macro'), zInd, iInd)
    with open(macroPath, "w") as mFile:
        mFile.write('StgMove({0}, {1}, {2}, 0);' #'OpenDocument("{0}",0);'
                    '\nLoadROI("{3}", 1)'.format(x,y,z,
                                                 imagePath.replace("/", "\\")))
    print("{0} has been created".format(macroPath))

########################################################################################################################

def cell_segmentation(nd2_path, coor_path, saveImages = False):
    #This is the meat of our script:

    #Getting current time:
    now = datetime.now()

    date = now.strftime("%Y-%m-%d %H%M%S")

    masksFolderPath = r"Masks ({})".format(date)

    try:
        os.mkdir(masksFolderPath)
    except:
        None #Folder already exists

    macroFolderPath = r"Macros ({})".format(date)
    try:
        os.mkdir(macroFolderPath)
    except:
        None #Folder already exists
    
    
    #Loading in the nd2 file
    nd2_file = nd2.ND2Reader(nd2_path)

    #Setting the FITC channel and finding out how many points there are in
    # the nd2 file:
    try:
        nd2_file.default_coords['c'] = 1
    except:
        #If there is one channel (the FITC) channel
        None

    try:
        # Defining the number of points for iteration:
        num_points = nd2_file.sizes['v']
    except:
        #If there is no property 'v' then we set the number of points to
        # 1:
        num_points = 1

    # Defining the number of z_stacks
    num_stacks = nd2_file.sizes['z']  # YK

    # Defining a list for all of our points: #YK
    # Defining a list for all of stack arrays
    z_stacks = []

    # We will iterate over the number of points taken
    for i in range(0, num_points):

        if num_points != 1:
            # For every loop, we will change the point.
            nd2_file.default_coords['v'] = i

        # Temporary list to store z-stack images post extraction
        z_stack_post = []

        # We will iterate over the number of #YK stacks
        for j in range(0, num_stacks):
            z_stack_post.append(np.asarray(nd2_file[j]))

        z_stacks.append(z_stack_post)

    z_stacks = normalizeZStacks(z_stacks)

    #If the user wants to save the z-stacks as images:
    if saveImages:
        org_images = z_stacks.copy()

    #Applying Median Filter:
    z_stacks = applyMedianFilter(z_stacks)

    #Applying threshold:
    z_stacks = applyTresholding(z_stacks)

    # Contains [BlobsListsPerImage[blobs[Blob2D]]]
    blobStacks = []  # YK

    # Minimum area size
    minAreaSize = 5000

    #Image index counts start at 1:
    imageIncInd = 1

    # We iterate like we've seen for the other sections. We loop through each
    # z-stack and then we iterate through each image.

    for zInd in range(0, len(z_stacks)):
        z_stack_th = z_stacks[zInd]

        # BlobImage lists i.e. all images' areas
        bStack = BlobStack(zstackInd=zInd)

        # iterating through each image. We use our median fitered image
        # to draw contours on.
        imageInd = 0

        for image_th in z_stack_th:
            # List of Blob2D i.e. areas in an image
            blobImage = BlobImage(incInd = imageIncInd)

            # Finding contours with function:
            contour, hierarchy = cv2.findContours(image_th, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_NONE)
            for c in contour:
                area = cv2.contourArea(c)

                # Filtering contours to an area larger than 4000.
                if area > minAreaSize:
                    # Calculating the centroid of each object
                    M = cv2.moments(c)
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])

                    blobImage.add(Blob2D(imageInd=imageInd,
                                         imageIncInd = imageIncInd,
                                         centroid=(cX, cY),
                                         contour=c,
                                         areaMembers=getAreaMembers(image=image_th,
                                                                    contour=c),
                                         radius=getFarthest(centroid=(cX, cY),
                                                            contour=c),
                                         area=area,
                                         zstackInd=zInd))
            imageInd += 1
            imageIncInd += 1

            # Finally storing all our data
            bStack.add(blobImage)  # YK

        blobStacks.append(bStack)

    # membership testing
    # Per stack
    for bStack in blobStacks:
        bImages = bStack.getBlobImages()
        for bImage in bImages:
            # lists grouped per images
            blobs = bImage.getBlobs()

            # blobs on a given image
            for bInd in range(0, len(blobs)):
                blob = blobs[bInd]
                masses = bStack.getMasses()
                if not isAMember(blob, masses):
                    mId = len(masses) + 1
                    m = Mass(mId=mId,
                             zstackInd=bStack.zstackInd)
                    m.add(blob)
                    bStack.addMass(mass=m)
        for m in bStack.getMasses():
            if not m.isSphere():
                continue
            m.setAlone(isAlone(masse=m, bImages=bImages))

    # The masses are available on the bStack object
    print('\n')
    print("Finding target masses in each z-stack...")

    coordinates = pd.read_excel(coor_path,sheet_name = 'Recorded Data', index_col=0,
                                usecols=['Index','X Coord [µm]','Y Coord [µm]','Ti ZDrive [µm]'])

    
    for bsInd in tqdm(range(0, len(blobStacks))):
        bStack = blobStacks[bsInd]
        print('z-stack {0} has {1} {2}'.format(bsInd + 1, bStack.numOfMasses(), 'masses'))
        for m in bStack.getMasses():

            if m.isAlone():  # m.isSphere()
                biggestBlob = m.getBiggestRadiusBlob()
                biggestBlob.zstackInd
                biggestBlob.imageInd
                biggestBlob.contour


                if m.isSphere() and biggestBlob.isSoliditaryElongation() and not (biggestBlob.isOnEdge()):
                    mask = biggestBlob.areaMembers

                    cv2.drawContours(mask, biggestBlob.contour, -1, (255, 0, 0), 3)

                    imagePath = join(
                        masksFolderPath,
                        'z{0}_image_{1}.png'.format(biggestBlob.zstackInd + 1, biggestBlob.imageInd + 1))

                    cv2.imwrite(imagePath, mask.astype('uint8'))

                    x = coordinates.loc[biggestBlob.imageIncInd,'X Coord [µm]']
                    y = coordinates.loc[biggestBlob.imageIncInd,'Y Coord [µm]']
                    z = coordinates.loc[biggestBlob.imageIncInd,'Ti ZDrive [µm]']

                    generateMacroFile(macroFolder = macroFolderPath,
                                      zInd = biggestBlob.zstackInd + 1,
                                      iInd = biggestBlob.imageInd + 1,
                                      nd2FilePath = nd2_path,
                                      imagePath = imagePath,
                                      x = x,
                                      y = y,
                                      z = z)

                # send those info to the microscope
                # shoot laser on it

    #         # for  accessing contained data
    #         for k, b in m.getBlobDict().items():

    #Saving images if user wants to:
    
    if saveImages:
        print('\n')
        print("Saving Images...")

        savedImg_path = r"Contours ({})".format(date)
        try:
            os.mkdir(savedImg_path)
        except:
            None #Folder already exists



        #Initiating list to store cell data (e.g. circularity, convexity, etc):
        z_r = []
        b_r = []
        i_r = []
        cir = []
        sol = []
        elong = []
        area = []
        j = 1
        
        for bsInd, z in zip(range(0, len(blobStacks)), org_images):
            bStack = blobStacks[bsInd]
            temp_zstack = z.copy()
            temp_zstack = [cv2.cvtColor(i, cv2.COLOR_GRAY2RGB) for i in temp_zstack]

            for m, blob_index in zip(bStack.getMasses(), range(1, bStack.numOfMasses() + 1)):
                biggestBlob = m.getBiggestRadiusBlob()

                for key, val in m.getBlobDict().items():

                    #Storing your data:
                    data = val.calculation()
                    z_r.append(j)
                    i_r.append(val.imageInd +1)
                    b_r.append(blob_index)
                    area.append(data[0])
                    cir.append(data[1])
                    sol.append(data[2])
                    elong.append(data[3])

                    # Creating an ROI mask from contours in the local directory

                    img_index = val.imageInd
                    c = val.contour

                    # Limiting the area to test if a point is in the contour
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = val.centroid
                    fontScale = 1
                    fontColor = (55, 55, 55)
                    lineType = 2

                    if m.isSphere() and biggestBlob.isSoliditaryElongation() and not (biggestBlob.isOnEdge()):

                        cv2.drawContours(temp_zstack[img_index], c, -1, (0, 255, 0), 3)

                        # Saving the
                        cv2.putText(temp_zstack[img_index], str(blob_index),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                    else:

                        cv2.drawContours(temp_zstack[img_index], c, -1, (0, 0, 255), 3)

                        # Putting number for objects in images:
                        cv2.putText(temp_zstack[img_index], str(blob_index),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

            j += 1


            #Cell data: 
            cell_data = pd.DataFrame({'b_ind': b_r, 'z_ind':z_r, 'area': area,
                                      'circularity':cir, 'solidity':sol, 'elongation':elong})

            cell_data.to_csv(join(savedImg_path, 'contour_data.csv'), index= False)
            
            #Saving each z-stack
            for i, ind in zip(temp_zstack, range(1, len(temp_zstack) + 1)):
                cv2.imwrite(join(savedImg_path, 'z{0}_image_{1}.png'.format((bsInd + 1), ind)), i)
            print("Contours has been saved in {}".format(savedImg_path))

########################################################################################################################

if __name__ == '__main__':
    #Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('nd2Path',
                        help = 'Path to ND2 File')
    parser.add_argument('coorPath',
                        help = 'Path to Coordinates')
    parser.add_argument('-o', '--saveImages',
                        help = "OPTIONAL boolean statement to save images with contours")
    args = parser.parse_args()

    badInputs = []

    for arg in [args.nd2Path, args.coorPath]:
        if not os.path.exists(arg):
            badInputs.append(arg)
        if len(badInputs) > 0:
            [print('{} does not exist! Check input file path'.format(x)) for x in badInputs]
            input("Press Enter to continue...")
            sys.exit()

    # Set output directory
    if args.saveImages in ["yes", "True", "true", "T", 1, "1", True]:
        args.saveImages = True

    elif args.saveImages in ["no", "False", "false", "t", 0, "0", False]:
        args.saveImages = False
    else:
        print('{} is not a boolean variable, so no contours will be saved!'.format(args.saveImages))
        args.saveImages = False
        input("Press Enter to continue...")
        

    

    #Cell segmentation:
    print("Cell Segmentation in Progress...")

    time_start = time.time()

    cell_segmentation(nd2_path = args.nd2Path,
                      coor_path = args.coorPath,
                      saveImages = args.saveImages)
    print('\n')    
    print("Total Elapsed: {}".format(time.time() - time_start))
    input("Press Enter to continue...")
