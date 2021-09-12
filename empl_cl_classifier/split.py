from os import listdir
from os.path import isfile, join
import shutil
from pathlib import Path


def main():
    path_employee_train = './dataset_empl_cl/train/employee/'
    path_employee_val = './dataset_empl_cl/val/employee/'
    path_client_train = './dataset_empl_cl/train/client/'
    path_client_val = './dataset_empl_cl/val/client/'

    create_req_dirs(path_client_train, path_employee_val, path_client_val, path_employee_train)

    dir_empl = './employee/'
    dir_client = './client/'
    files_empl = [f for f in listdir(dir_empl) if (isfile(join(dir_empl, f)))]
    files_cl = [f for f in listdir(dir_client) if (isfile(join(dir_client, f)))]

    counter = 0
    for file_empl, file_client in zip(files_empl, files_cl):

        if counter % 5 == 0:
            shutil.copy(dir_empl + file_empl, path_employee_val + file_empl)
            shutil.copy(dir_client + file_client, path_client_val + file_client)

        else:
            shutil.copy(dir_empl + file_empl, path_employee_train + file_empl)
            shutil.copy(dir_client + file_client, path_client_train + file_client)

        counter += 1


def create_req_dirs(*args):
    for arg in args:
        Path(arg).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main()

