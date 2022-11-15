import numpy as np
import multiprocessing as mp
import yaml
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.utils import Sequence


from Code.Classes.Image import Image


class ImageLoader(Sequence):

    """Read images from path, storing them in Image class"""

    def __init__(
        self,
        data_paths,
        labels,
        set_type="train",
        batch_size=32,
        maximum=None,
        minimum=None,
    ):

        """Constructor for ImageLoader class"""

        # Initiate pool for multiprocessing (for speeding up reading of images)
        pool = mp.Pool(mp.cpu_count())

        # Load images
        data = np.array(pool.map(Image.from_path, tqdm(data_paths)))

        # Close pool
        pool.close()

        # Store details
        self.data_paths = np.array(data_paths)

        self.data = data
        self.labels = labels

        self.batch_size = batch_size
        self.number = len(data_paths)
        self.names = [Path(path).stem for path in self.data_paths]
        self.set_type = set_type

        self.set_shape()

        # Set maximum and minimum if not fixed
        if maximum == None or minimum == None:
            self.maximum, self.minimum = self.get_extrema()
        else:
            self.maximum, self.minimum = maximum, minimum

    def __len__(self):

        """Return the number of batches needed for completing the dataset"""

        return int(np.ceil(self.number / float(self.batch_size)))

    def __getitem__(self, i):

        """Return the images in the i-th batch"""

        # Find paths for the i-th batch
        data_batch = self.data[i * self.batch_size : (i + 1) * self.batch_size]
        label_batch = np.array(
            [
                self.labels[n].label
                for n in range(i * self.batch_size, (i + 1) * self.batch_size)
            ]
        )

        # Normalise images
        normalised_images = [
            Image.normalise(image, self.maximum, self.minimum) for image in data_batch
        ]

        # Retrieve images in the form of tensor from the batch
        data_batch = np.array(
            [normalised_image.tensor for normalised_image in normalised_images]
        )

        return data_batch, np.array(label_batch)

    def set_shape(self):

        """Check that all images in dataset have the same shape. If not, raise an error"""

        # Retrieve shapes (specify that it should be a numpy array of tuples)
        shapes = np.array([tensor.shape for tensor in list(self.data)], dtype="i,i")

        # Check unicity
        shape = np.unique(shapes)

        # If more than one value
        if shape.ndim > 1:

            raise ValueError("Images in dataset do not have the same shape")

        self.shape = shape[0]

    def get_extrema(self):

        """Get maximum and minimum for all images in dataset"""

        # Get maximum and minimum
        maximum = np.max(np.array([image.tensor for image in self.data]))
        minimum = np.min(np.array([image.tensor for image in self.data]))

        return maximum, minimum

    def save_split(self):

        """Save the splitting of data"""

        with open(f"Split/{self.set_type}.yaml", "w") as file:

            split = {"files": self.names}

            yaml.dump(split, file)

    def split(self, indices):

        dataset = ImageLoader(
            self.data_paths[indices],
            self.set_type,
            self.batch_size,
            self.maximum,
            self.minimum,
        )

        return dataset
