import os
import tensorflow as tf


class DataLoader:
    def __init__(self, train_dir=None, val_dir=None, img_size=224, batch_size=32, seed=42):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed

    def _map_to_3_classes(self, alm_idx, healthy_idx):
        def map_fn(img, label):
            new_label = tf.fill(tf.shape(label), 2)
            new_label = tf.where(tf.equal(label, alm_idx), tf.constant(0, dtype=tf.int32), new_label)
            new_label = tf.where(tf.equal(label, healthy_idx), tf.constant(1, dtype=tf.int32), new_label)
            return img, new_label

        return map_fn

    def load_data(self):
        if self.train_dir is None or self.val_dir is None:
            raise ValueError("train_dir and val_dir must be provided")

        if not os.path.exists(self.train_dir) or not os.path.exists(self.val_dir):
            raise FileNotFoundError("Training or validation directories do not exist")

        raw_train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            seed=self.seed,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
        )

        raw_val_ds = tf.keras.utils.image_dataset_from_directory(
            self.val_dir,
            seed=self.seed,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
        )

        original_classes = raw_train_ds.class_names
        alm_idx = original_classes.index("Acral_Lentiginous_Melanoma")
        healthy_idx = original_classes.index("Healthy_Nail")

        train_ds = raw_train_ds.map(self._map_to_3_classes(alm_idx, healthy_idx), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = raw_val_ds.map(self._map_to_3_classes(alm_idx, healthy_idx), num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        class_names = ["Acral_Melanoma", "Healthy_Nail", "Other_Condition"]
        return train_ds, val_ds, class_names


if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_data()
    print("✅ Data loaded", data[2])
