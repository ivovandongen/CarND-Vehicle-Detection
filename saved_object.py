import pickle
import os.path

from tracer_decorator import traced


class SavedObject:
    """
    Base for classes that need to be saved to disk
    """

    @staticmethod
    @traced
    def _load(file_name):
        if not os.path.isfile(file_name):
            print("No such file", file_name)
            return None

        with open(file_name, 'rb') as f:
            print("Loading", file_name)
            return pickle.load(f)

    @traced
    def _save(self, file_name):
        with open(file_name, 'wb') as f:
            print("Saving", file_name)
            pickle.dump(self, f)

    @staticmethod
    def _create(create_instance_f, save_file_name, create_instance_args=None):
        save_file_name = 'saved_state/' + save_file_name
        obj = None
        try:
            obj = SavedObject._load(save_file_name)
        except Exception as e:
            print("Could not load", save_file_name, ":", e)

        if obj is None:
            obj = create_instance_f() if create_instance_args is None else create_instance_f(**create_instance_args)
            obj._save(save_file_name)

        return obj
