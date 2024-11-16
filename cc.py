import numpy
import cv2
def undistort(image):
    im = numpy.loadtxt('intrinsic.txt')
    dc = numpy.loadtxt('dist.txt')

    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(im, dc, (w, h), 1, (w, h))
    x, y, w, h = roi
    return cv2.undistort(image, im, dc, None, newcameramtx)[y:y + h, x:x + w], cv2.undistort(image, im, dc, None,
                                                                                             newcameramtx)
path = './res/5.jpg'
img1,img2 = undistort(cv2.imread(path))
img3 = cv2.imread(path)

cv2.imshow("og",img3)
cv2.imshow("undistorted",img1)

cv2.waitKey(10000)
cv2.destroyAllWindows()