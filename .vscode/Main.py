import cv2
import numpy as np

def main():
    # Meminta pengguna untuk memasukkan nama file video
    video_path = input("Masukkan path file video: ")
    output_filename = input("Masukkan nama file output (termasuk ekstensi): ")

    # Menginisialisasi video
    cap = cv2.VideoCapture(video_path)

    # Memeriksa keberhasilan pembacaan video
    if not cap.isOpened():
        print("Gagal membaca video atau video tidak ditemukan")
        return

    # Mengambil ukuran jendela awal
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_factor = 0.8  # Faktor skala untuk mengubah ukuran tampilan

    # Inisialisasi model MOG2 untuk pengurangan latar belakang
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Menginisialisasi video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_filename, fourcc, 20.0, (int(width * scale_factor), int(height * scale_factor)))

    while True:
        # Membaca frame
        ret, frame = cap.read()

        # Keluar dari loop jika pembacaan frame gagal
        if not ret:
            break

        # Mengubah ukuran frame
        frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

        # Mengurangi latar belakang menggunakan model MOG2
        fg_mask = background_subtractor.apply(frame)

        # Menyaring noise dengan operasi morfologi
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Menemukan kontur objek yang bergerak
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Menyoroti area yang bergerak dengan kotak pembatas
        for contour in contours:
            if cv2.contourArea(contour) > 2000:  # Ubah nilai ambang batas sesuai kebutuhan
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Menampilkan frame
        cv2.imshow('Motion Detection', frame)

        # Menulis frame ke video output
        output_video.write(frame)

        # Menghentikan program jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Melepaskan sumber daya
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    # Menampilkan pilihan untuk memulai ulang atau keluar
    while True:
        choice = input("Video telah selesai. Apakah Anda ingin memulai ulang (y/n)? ")

        if choice.lower() == 'y':
            main()  # Memanggil fungsi main untuk memulai ulang deteksi gerakan
            break
        elif choice.lower() == 'n':
            print("Program selesai.")
            return
        else:
            print("Pilihan tidak valid. Silakan masukkan 'y' untuk memulai ulang atau 'n' untuk keluar.")

if __name__ == '__main__':
    main()
