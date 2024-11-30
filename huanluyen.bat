@echo off
echo.
echo =======================================================
echo          BAT DAU KIEM TRA DU LIEU
echo =======================================================
echo Dang chay code kiem tra du lieu (kiem tra cac gia tri thieu, du lieu khong hop le)...
echo =======================================================
python 1_kiemtradulieu.py
echo =======================================================
echo Kiem tra du lieu hoan tat. Nhan Enter de tiep tuc...
echo =======================================================
pause

@echo off
echo.
echo =======================================================
echo          BAT DAU TIEN XU LY DU LIEU
echo =======================================================
echo Dang chay code tien xu ly du lieu (chuan hoa, loai bo stopwords, v.v.)...
echo =======================================================
python 2_tienxulydulieu.py
echo =======================================================
echo Tien xu ly du lieu hoan tat. 
echo Ban có the kiem tra anh tienxuly_visual.png trong thu muc "Visual" de biet them chi tiet
echo =======================================================
echo Tien xu ly du lieu hoan tat. Nhan Enter de tiep tuc...
echo =======================================================
pause

@echo off
echo.
echo =======================================================
echo          BAT DAU PHAN TACH DU LIEU
echo =======================================================
echo Dang chay code phan tach du lieu (chia du lieu thanh tap huan luyen va kiem tra)...
echo =======================================================
python 3_phantachdulieu.py
echo =======================================================
echo Phan tach du lieu hoan tat. Nhan Enter de tiep tuc...
echo =======================================================
pause

@echo off
echo.
echo =======================================================
echo          BAT DAU HUAN LUYEN MO HINH
echo =======================================================
echo Dang chay code huan luyen du lieu (huan luyen mo hinh hoc may)...
echo =======================================================
python 4_huanluyendulieu.py
echo =======================================================
echo Ban có the kiem tra anh huanluyen_confusion_matrix.png trong thu muc "Visual" de biet them chi tiet
echo =======================================================
echo Huan luyen mo hinh hoan tat. Nhan Enter de tiep tuc...
echo =======================================================
pause

@echo off
echo.
echo =======================================================
echo          BAT DAU KIEM TRA MO HINH
echo =======================================================
echo Dang chay code kiem tra mo hinh (kiem tra do chinh xac, confusion matrix, v.v.)...
echo =======================================================
python 5_kiemtramohinh.py
echo =======================================================
echo Kiem tra mo hinh hoan tat. Nhan Enter de tiep tuc...
echo =======================================================
pause

@echo off
echo.
echo =======================================================
echo          QUY TRINH HOAN TAT
echo =======================================================
echo Qua trinh huan luyen mo hinh da hoan tat thanh cong.
echo Ban co the su dung mo hinh da huan luyen trong thu muc "Model".
echo Hay chay lai file run.bat neu ban muon su dung mo hinh.
echo =======================================================
pause
