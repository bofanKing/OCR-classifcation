import sys
from PyQt5.QtWidgets import QApplication
from User import ResNetUserApp, MobileNetUserApp

"""
模型结果展示：
ResNet50 Accuracy: 0.9326 | F1-score: 0.9284
                          precision    recall  f1-score   support
registration_certificate     0.9327    0.9123    0.9224      1003
         driving_license     0.9329    0.9369    0.9349       935
          passport_files     1.0000    0.9048    0.9500        42
          contract_files     0.8807    0.8935    0.8871      3372
           invoice_files     0.9222    0.9345    0.9283      3372
          vehicle_photos     0.9995    0.9995    0.9995      5502
           receipt_files     0.8857    0.8672    0.8764      3450
                accuracy                         0.9326     17676
               macro avg     0.9362    0.9212    0.9284     17676
            weighted avg     0.9326    0.9326    0.9325     17676

MobileNet Accuracy: 0.9365 | F1-score: 0.9347
                          precision    recall  f1-score   support
registration_certificate     0.9603    0.8933    0.9256      1003
         driving_license     0.9523    0.9390    0.9456       935
          passport_files     1.0000    0.9286    0.9630        42
          contract_files     0.8355    0.9413    0.8852      3372
           invoice_files     0.9465    0.9291    0.9377      3372
          vehicle_photos     0.9998    0.9996    0.9997      5502
           receipt_files     0.9247    0.8501    0.8858      3450
                accuracy                         0.9365     17676
               macro avg     0.9456    0.9259    0.9347     17676
            weighted avg     0.9389    0.9365    0.9367     17676
"""


SELECTED_MODEL = "mobilenet" # 性能更好

def main():
    app = QApplication(sys.argv)  #
    if SELECTED_MODEL == "resnet":
        print("使用 ResNet50 推理与 GUI")
        model = ResNetUserApp()
        model.run_inference()
        model.launch_gui()

    elif SELECTED_MODEL == "mobilenet":
        print("使用 MobileNetV2 推理与 GUI")
        model = MobileNetUserApp()
        model.run_inference()
        model.launch_gui()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
