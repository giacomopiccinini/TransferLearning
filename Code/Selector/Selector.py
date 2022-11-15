from importlib import import_module


class Selector:

    """Class used to select various options, e.g. optimisers or loss functions"""

    def __init__(self, object_type):

        """Constructor. Must specify the objects we will be selecting"""

        # Store object type
        self.object_type = object_type.lower()

        # Store relevant directory
        self.directory = f"Code.{object_type.lower().capitalize()}"

    def select(self, object):

        """Load element"""

        try:
            # Import module for standard organisation of Coding directory
            module = import_module(f"{self.directory}.{object}")

            # Import function from module
            function = getattr(module, self.object_type)

            return function

        except Exception as e:
            # Raise exception
            raise e
