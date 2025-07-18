class CFG:
    num_classes = 7
    batch_size = 32
    num_epochs = 10
    patience = 3  # early stopping
    lr = 1e-4
    image_size = 224
    device = 'cuda'  # or 'cuda:0'
    train_txt = 'train.txt'
    val_txt = 'val.txt'
    test_txt = 'test.txt'

label_map = {
    "registration_certificate": 0,
    "driving_license": 1,
    "passport_files": 2,
    "contract_files": 3,
    "invoice_files": 4,
    "vehicle_photos": 5,
    "receipt_files": 6,
}
