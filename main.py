import logging
import os
from pathlib import Path

from Datasets.data_loader.A_data_loader import ADataLoader
from Datasets.data_loader.B_data_loader import BDataLoader

from A1.A1 import A1
from A2.A2 import A2
from B1.B1 import B1
from B2.B2 import B2


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


root = get_project_root()

logger = logging.getLogger(__name__)


class TestModel:
    """
    Test to see keras model works correctly
    """

    def test_model_A1(self):
        task = "gender"
        data = ADataLoader(X_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/celeba/img/'),
                           Y_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/celeba/labels.csv'),
                           task=task)

        x_train, y_train = data.get_train_data()
        x_test, y_test = data.get_test_data()

        input_dim = x_train.shape[1:]

        model_A1 = A1(input_dim, x_train, y_train, x_test, y_test)
        model_A1.train()
        _, acc_A1_train = model_A1.evaluate_model(task)

        return acc_A1_train

    def test_model_A2(self):
        task = "smiling"
        data = ADataLoader(X_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/celeba/img/'),
                           Y_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/celeba/labels.csv'),
                           task=task)

        x_train, y_train = data.get_train_data()
        x_test, y_test = data.get_test_data()

        input_dim = x_train.shape[1:]

        model_A2 = A2(input_dim, x_train, y_train, x_test, y_test)
        model_A2.train()
        _, acc_A2_train = model_A2.evaluate_model(task)

        return acc_A2_train

    def test_model_B1(self):
        task = "face_shape"
        data = BDataLoader(X_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/cartoon_set/img/'),
                           Y_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/cartoon_set/labels.csv'),
                           task=task)

        x_train, y_train = data.get_train_data()
        x_test, y_test = data.get_test_data()

        input_dim = x_train.shape[1:]

        model_B1 = B1(input_dim, x_train, y_train, x_test, y_test)
        model_B1.train()
        _, acc_B1_train = model_B1.evaluate_model(task)

        return acc_B1_train

    def test_model_B2(self):
        task = "eye_color"
        data = BDataLoader(X_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/cartoon_set/img/'),
                           Y_data_path=os.path.join(str(root), 'AMLS_assignment20_21/Datasets/cartoon_set/labels.csv'),
                           task=task)

        x_train, y_train = data.get_train_data()
        x_test, y_test = data.get_test_data()

        input_dim = x_train.shape[1:]

        model_B2 = B2(input_dim, x_train, y_train, x_test, y_test)
        model_B2.train()
        _, acc_B2_train = model_B2.evaluate_model(task)

        return acc_B2_train


if __name__ == '__main__':
    test = TestModel()
    acc_A1 = test.test_model_A1()
    acc_A2 = test.test_model_A2()
    acc_B1 = test.test_model_B1()
    acc_B2 = test.test_model_B2()

    print('TA1: {} - '
          'TA2: {} - '
          'TB1: {} - '
          'TB2: {}  '.format(acc_A1,
                             acc_A2,
                             acc_B1,
                             acc_B2, ))
