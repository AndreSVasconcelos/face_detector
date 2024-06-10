# Libs
import cv2 # OpenCV

# Lendo imagem
imagem = cv2.imread("./images/group.jpg")
#cv2.imshow("Imagem", imagem)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Convertr a imagem para tons de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Imagem", imagem_cinza)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Carregar cascade
detector = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

# Detectar faces
faces = detector.detectMultiScale(imagem_cinza, 
                                  scaleFactor=1.6, 
                                  minNeighbors=5,
                                  minSize=(30, 30))
print(faces)

# Desenhar retangulo
for (x, y, l, a) in faces:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
cv2.imshow("Imagem", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()