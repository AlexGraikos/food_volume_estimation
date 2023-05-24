from load_food_rec import CocoDatasetGenerator
from food_recognition_options import FoodRecognitionOptions


def train(options: FoodRecognitionOptions):
    train_generator = CocoDatasetGenerator(annotations_path=options.train_ann_path, img_dir=options.train_img_path, data_size=options.data_size)
    val_generator = CocoDatasetGenerator(annotations_path=options.val_ann_path, img_dir=options.val_img_path, data_size=options.data_size)
    
    t_gen = train_generator.createDataGenerator(batch_size=options.batch_size)
    v_gen = val_generator.createDataGenerator(batch_size=options.batch_size)

    train_generator.visualizeGenerator(t_gen)

    for i in range(options.epochs):
        img_batch, mask_batch = next(t_gen)

if __name__ == "__main__":
    train(FoodRecognitionOptions())



