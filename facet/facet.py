import matplotlib
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("facet/Albert.jpg")
image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)

plt.imshow(image)
model = cv2.CascadeClassifier("facet/model.xml")
facet = model.detectMultiScale(image)

a = facet[0][0]
b = facet[0][1]
c = facet[0][2]
d = facet[0][3]

image = cv2.rectangle( image , (a,b) , (a+c,b+d) , (0,255,255) , 3 )

