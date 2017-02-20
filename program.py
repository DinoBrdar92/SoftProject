import numpy as np
import cv2
from sklearn.externals import joblib
from skimage.feature import hog

#FUNCTIONS

# Sluzi za izvlacenje linije sa slike (filter za plavu boju)
def filter_line(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    for i in np.arange(0, img_rgb.shape[0]):
        for j in np.arange(0, img_rgb.shape[1]):
            if \
                                            img_rgb[i, j, 2] < 50 and \
                                            img_rgb[i, j, 1] < 50 and \
                            not img_rgb[i, j, 0] < 150:
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    img_gray = img_gray.astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    img_gray_dillated = cv2.dilate(img_gray, kernel, iterations=2)

    return img_gray_dillated

# Sluzi za izvlacenje broja sa slike (filter za belu boju)
def filter_number(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    whiteness = 154
    for i in np.arange(0, img_rgb.shape[0]):
        for j in np.arange(0, img_rgb.shape[1]):
            if img_rgb[i, j, 0] > whiteness and img_rgb[i, j, 1] > whiteness and img_rgb[i, j, 2] > whiteness:
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    img_gray = img_gray.astype('uint8')

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img_gray_eroded = cv2.erode(img_gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    img_gray_dillated = cv2.dilate(img_gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)

    return img_gray_dillated

# Sluzi za pronalazenje koeficijenata linearne jednacine na osnovu dve prosledjene tacke
def equation_line(x1, y1, x2, y2):
    k=0.0
    y22 = y2+0.000
    y11 = y1+0.000
    k = (y22-y11)/(x2-x1)
    n = k*(-x1) + y1
    return [k, n]

# Proverava preklapanje izmedju linije i tacke
def check_overlapping(k, n, x, y):
    yy=k*x+n
    if abs(yy-y)<1:
        return True
    else:
        return False

#MAIN

# ucitavanje baze rukopisa
clf = joblib.load("digits_cls.pkl")

# ucitavanje test frejma - ovde ce trebati da se umetne logika za ucitavanje videa i ekstraktovanje frejmova
# frame = cv2.imread("frame.jpg")

file = open('outx.txt', 'w')
file.close()
final_sum = 0

for p in np.arange(0, 10):  # Ovaj for ti sluzi da prodje kroz 10 videa

    cap = cv2.VideoCapture("VideosTest/video-" + str(p) + ".avi")  # Ucitavanje videa
    # br_frame = 1.0  # Kretanje od certain frame-a
    # ret, frame = cap.read()
    # if ret == False:
    #     break
    #     br_frame = br_frame + 1
    # cap.release()
    # cap = cv2.VideoCapture("VideosTest/video-" + str(p) + ".avi")

    x = 0;
    cap.set(1, x);
    while True:
        x = x + 1
        ret, frame = cap.read()
        if ret == False:
            break


        # inicijalizacija parametara jednacine prave
        k11 = 0
        n11 = 0
        k0 = 0
        n0 = 0
        k22 = 0
        n22 = 0
        k2 = 0
        n2 = 0

        # sum = []

        # operacije nad linijom (treba da je samo za prvi frejm)
        im_line = filter_line(frame)                                                                       # izvlacenje linije sa slike
        _, ctrs_line, _ = cv2.findContours(im_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # trazenje konture od linije
        rects_line = [cv2.boundingRect(ctr) for ctr in ctrs_line]                                       # izvlacenje pravougaonog okvira na osnovu konture
        rect_line = rects_line[0]   # izvlacenje jedinog pravougaonika
        h_line = rect_line[3]       # visina linije
        w_line = rect_line[2]       # sirina linije

        [k0, n0] = equation_line(rect_line[0], 480 - rect_line[1] - rect_line[3], rect_line[0] + rect_line[2], 480 - rect_line[1])  # trazenje koeficijenata jednacine prave
        ww = (rect_line[0], rect_line[0] + rect_line[2])    # maksimalna i minimalna sirina linije (ZAMENITI MESTA?)

        # cv2.rectangle(frame, (rect_line[0], rect_line[1]), (rect_line[0] + rect_line[2], rect_line[1] + rect_line[3]), (255, 0, 0), 2)


        # operacije nad brojevima (treba da je za svaki frejm)
        im_number = filter_number(frame)                                                                       # izvlacenje brojeva sa slike
        _, ctrs_number, _ = cv2.findContours(im_number.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # trazenje konture od brojeva
        rects_number = [cv2.boundingRect(ctr) for ctr in ctrs_number]                                       # izvlacenje pravougaonih okvira na osnovu brojeva

        for rect in rects_number:  # Prolazim kroz regione za brojeve
            h = rect[3]  # visina regiona broja
            w = rect[2]  # sirina regiona broja

            # centar regiona broja
            x_c = rect[0] + rect[2] / 2
            y_c = 480 - (rect[1] + rect[3] / 2)

            rect_center = (x_c, y_c)


            # Draw the rectangles
            # cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 255), 2)
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            # x = true_value if condition else false_value
            pt1 = 0 if pt1 < 0 else pt1
            pt2 = 0 if pt2 < 0 else pt2

            roi = im_number[pt1:pt1 + leng, pt2:pt2 + leng]
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

            if  ww[0] < x_c < ww[1] and check_overlapping(k0, n0, x_c, y_c): #or check_overlapping(k1, n1, x_c, y_c)): # Provera da li se centar regiona od broja nalazi na jednoj od one dve prave
                # cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
                final_sum += int(nbr)
                x = x + 2

            # else:
            #     cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

            # sum.append(int(nbr))    #dodavanje na listu
            # cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        # cv2.imshow("Resulting Image with Rectangular ROIs", frame)
        # cv2.waitKey()
        # print sum
        print 'suma: ' + str(final_sum) + ' -- frame: ' + str(x) +  ' ~' + str(x / 40.0) + ' sec'


    file = open('outx.txt', 'a')
    file.write("video-" + str(p) + ".avi" + '\t' + str(final_sum) + '\n')
    file.close()
    cap.release()