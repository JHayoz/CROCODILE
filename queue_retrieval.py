import os
path_config_file='/home/ipa/quanz/user_accounts/jhayoz/Projects/CO_ratio_snowlines/retrievals/aflepb/aflepb_data_v01_test_no_hirise'
retrieval_path = '/home/jhayoz/Projects/CROCODILE/retrieval.py'
os.system('nice -n19 mpiexec -n 30 python %s %s' % (retrieval_path,path_config_file))