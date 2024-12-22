import cv2
import numpy as np

# Path ke video
cap = cv2.VideoCapture(r"C:\Pengolahan Citra\CircleDetection\WhatsApp Video 2024-12-20 at 11.37.30_891aff04.mp4")

if not cap.isOpened():
    print("Tidak dapat membuka video. Periksa kembali path video.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Video selesai atau frame tidak dapat dibaca.")
        break
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur untuk mengurangi noise
    gray_blurred = cv2.medianBlur(gray, 7)
    
    # Deteksi lingkaran menggunakan HoughCircles
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=100, 
        param2=30, 
        minRadius=10, 
        maxRadius=100
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            x, y, radius = circle
            
            # Validasi circularity
            mask = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)
            contour_area = cv2.countNonZero(mask)
            expected_area = np.pi * (radius ** 2)
            circularity = contour_area / expected_area
            
            if 0.9 <= circularity <= 1.1:
                # Gambar lingkaran pada frame
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), 3)
    
    # Tampilkan frame dengan deteksi lingkaran
    cv2.imshow("Circle Detection", frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
