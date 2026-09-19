import os
import time
import sys
import argparse
from A1001_Find_Max_Batchsize import Pattern as find_maxbs
from A1002_Common_Model_Check import Pattern as model_check

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from products.aiDAPTIVLink.pattern import Get_tookit_version
from products.aiDAPTIVLink.argument import MW_Argument
from products.aiDAPTIVLink.utils import get_project_ini

def replace_argument(argument: MW_Argument, project_ini) -> MW_Argument:
    argument.specify_gpus = project_ini["ENV_setting"]["specify_gpu_index"]
    argument.num_gpus = int(project_ini["ENV_setting"]["num_gpus"])
    argument.model_name_or_path = project_ini["ENV_setting"]["model_name_or_path"]
    argument.nvme_path = project_ini["ENV_setting"]["nvme_path"]
    argument.max_seq_len = int(project_ini["Performance_test"]["seq_len"])
    argument.triton = project_ini["Performance_test"]["triton"] == 'True'
    return argument

def model_run(find=True, performance=True):
    # [create argument and replaced by ini]
    argu = MW_Argument()
    ini = get_project_ini(os.path.join(os.path.dirname(__file__).split("Script/")[0], "project.ini"))
    argu = replace_argument(argu, ini)
    start_bs = int(ini['Performance_test']['start_bs'])
    end_bs = int(ini['Performance_test']['end_bs'])
    cmd_timeout = float(ini['Performance_test']['training_hour'])

    if find:
        find = find_maxbs(argu, start=start_bs, end=end_bs)
        max_batchsize = find()
        if max_batchsize == -1:
            print("The start batchsize is too high.")
            return
        elif max_batchsize == -2:
            print("Training fail.")
            return
        print(f"max_batchsize={max_batchsize}")
        argu.per_device_train_batch_size = max_batchsize
    else:
        argu.per_device_train_batch_size = int(ini["Performance_test"]["start_bs"])
    
    time.sleep(10)

    if performance:
        training = model_check(argu, cmd_timeout)
        training()
        time.sleep(10)

if __name__ == "__main__":

    test_description = "Choose aiDAPTIV Toolkit test item:\n1. Performance Test\n2. Test max batchsize\n3. Find max batchsize then test performance"

    parser = argparse.ArgumentParser() 
    parser.add_argument('-t', '--test', dest='test', help=f"{test_description}", default='0', type=int)
    parser.add_argument('-v', '--version', help="aiDAPTIV Toolkit Version", action="store_true")
    argsss = parser.parse_known_args()[0]

    if argsss.version:
        print(Get_tookit_version())
    elif argsss.test == 1:
        print("choice 1: Model Performance Test")
        model_run(find=False, performance=True)
    elif argsss.test == 2:
        print("choice 2: Test max batchsize")
        model_run(find=True, performance=False)
    elif argsss.test == 3:
        print("choice 3: Find max batchsize then test performance")
        model_run(find=True, performance=True)
    else:
        print("Invalid choice parameter.")