import os

from ami.settings import BASE_DIR

def delect_temp_file(dir):
    if dir == 'prediction':
        file_list = os.listdir(BASE_DIR / 'model' / 'inputs' / dir)
        for file in file_list:
            if file[0] == '.':
                continue
            file_path = BASE_DIR / 'model' / 'inputs' / dir / file
            os.remove(file_path)

    elif dir == 'plots':
        file_list = os.listdir(BASE_DIR / 'static' / dir)
        for file in file_list:
            if file[0] == '.' or file == 'confusion_matrix.png' or file == 'dataloader_dist.png':
                continue
            file_path = BASE_DIR / 'static' / dir / file
            os.remove(file_path)

    elif dir == 'plots/conf':
        file_list = os.listdir(BASE_DIR / 'static' / 'plots')
        for file in file_list:
            if file[0] == '.' or file == 'confusion_matrix.png' or file == 'dataloader_dist.png':
                continue
            file_path = BASE_DIR / 'static' / 'plots' / file
            os.remove(file_path)

    elif dir == 'temp':
        file_list = os.listdir(BASE_DIR / 'model' / 'inputs' / 'a_learning' / dir)
        for file in file_list:
            if file[0] == '.':
                continue
            file_path = BASE_DIR / 'model' / 'inputs' / 'a_learning' / dir / file
            os.remove(file_path)