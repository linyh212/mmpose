### **YOLO 訓練 / 資料集的問題說明**

1. **舊的資料集設計錯誤**
   - 一開始我用的是 /home/divclab/group1/dragonboat_yolo/images 裡「已經畫好框的圖片」去訓練 YOLO。
   - 也就是影像裡同時有：原始畫面 + RF-DETR 畫出來的藍框和文字。
   - 這樣 YOLO 會學到的是「藍色框＋文字的外觀」，而不是單純學習「槳本身的外觀」，導致：
     - 在有藍框的圖片上表現很好
     - 換成乾淨的影片 / 圖片時幾乎偵測不到槳。

2. **中途有手動刪除過一部分 pseudo label**
   - Predict_Output/*/images、labels 裡有部分檔案被我刪掉過。
   - 所以舊的 dragonboat_yolo 目錄裡，**image / label 數量不一定一致**，而且命名方式也不好（A、B、C 混在一起）。

3. **目前的處理方向**

   - 刪除 dragonboat_yolo/，改用「raw video + RF-DETR label」重新建立一個乾淨的 YOLO 資料集。

   - 具體做法一：

     1. 每支影片 Source_DataSet/videos/A04.mp4 對應到 RF-DETR 的輸出 Predict_Output/A04/labels/*.txt。

     2. 先讀 labels 的檔名，例如 frame_000407.txt，把 000407 這個 frame index 記下來。

     3. 從 raw video A04.mp4 重新從 frame 0 開始一幀一幀讀，讀到第 407 幀時，把「當下這一幀原始畫面」存成

        images/train/A04_frame_000407.jpg，並把對應的 frame_000407.txt 複製成labels/train/A04_frame_000407.txt。

     4. 以此類推，對所有影片 / 所有現存的 label 做同樣處理。

     - 這樣可以確保：
       - 圖片是**沒有畫框的原始 frame**，
       - label 仍然**沿用** RF-DETR 當初輸出的座標，
       - image / label 透過 frame index 對起來。

     

