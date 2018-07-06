import pickle
import os.path


class SavedObject:
    """
    Base for classes that need to be saved to disk
    """

    @staticmethod
    def _load(file_name):
        if not os.path.isfile(file_name):
            print("No such file", file_name)
            return None

        with open(file_name, 'rb') as f:
            print("Loading", file_name)
            return pickle.load(f)

    def _save(self, file_name):
        with open(file_name, 'wb') as f:
            print("Saving", file_name)
            pickle.dump(self, f)

    @staticmethod
    def _create(create_instance_f, save_file_name):
        obj = SavedObject._load(save_file_name)

        if obj is None:
            obj = create_instance_f()
            obj._save(save_file_name)

        return obj
