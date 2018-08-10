'''
Created on Jul 3, 2018

@author: josephnunn
'''
import os
import re
import subprocess as sub
import threading as thrd

test_pattern = re.compile('test_.*\.py')
test_dir = ''
py_exec = 'python'
halt = 5
#
# def all_tests():
#     tests_failed = 0
#     dir = os.getcwd() + test_dir
#     test_list = [x for x in os.listdir(dir) if test_pattern.search(x) ]
#     print(test_list)
#     print('after list')
#
#     name = "path"
#
#     def test_helper(test):
#         result = sub.run(['%s' % py_exec, '%s' % test], timeout=20)
#         return result.returncode
#
#     for test_file in test_list:
#         print('test file')
#         print(test_file)
#         print('after test file')
#         try:
#             thrd.Thread(target=test_helper(test_file)).start()
#         except:
#             print("Error all_tests:all_tests() test %s failed" % test_file)
#             tests_failed += 1
#
# #         print(os.getcwd())
# # #         result = sub.run('python ./%s' % test_file ).returncode()
# #         result = sub.run(['%s' % py_exec, '%s' % test_file], timeout=20)
# # #        result = sub.run('/bin/ls', timeout=2)
# #         code = result.returncode
# #         print(code)
# #         if result:
# #             print("Error all_tests:all_tests() test %s failed" % test_file)
# #             tests_failed += 1
#
# #         try:
# # #            test_mod = importlib.abc.MetaPathFinder.find_module( test_file )
# #             result = sub.run('python %s' % test_file ).returncode()
# # #           test = importlib.import_module( test_file )
# # #           test.run_test()
# #         except:
# #             print("Error all_tests:all_tests() test %s failed" % test_file)
# #             tests_failed += 1
# #
#     return tests_failed
#


def all_tests():
    tests_failed = 0
    dir = os.getcwd() + test_dir
    test_list = [x for x in os.listdir(dir) if test_pattern.search(x)]
    print(test_list)
    print('after list')

    name = "path"
    proc_list = []

#     def test_helper(test):
#         result = sub.Popen(['%s' % py_exec, '%s' % test], timeout=20)
#         return result.returncode

#     def test_helper(test):
#         proc = sub.Popen(['%s' % py_exec, '%s' % test], timeout=20)
#
#         try:
#
#         except TimeoutExpired:
#             proc.kill()

    for test_file in test_list:
        print('Running test %s' % test_file)
        try:
            proc_list.append(sub.Popen(['%s' % py_exec, '%s' % test_file]))

#             test_helper(test_file)

            # thrd.Thread(target=test_helper(test_file)).start()
        except:
            print("Error all_tests:all_tests() test %s failed" % test_file)
            tests_failed += 1

    guard = True
    while guard:
        guard = False
        for proc in proc_list:
            if (proc.poll() is None):
                guard = True

    failed_list = []
    index = 0
    for proc in proc_list:
        if (proc.poll() != 0):
            tests_failed += 1
            failed_list.append(test_list[index])
        index += 1

    return (tests_failed, failed_list)

if __name__ == '__main__':
    print('Running all_test:main()')
    num_failed, tests_failed = all_tests()
    print('all_tests run')
    print('%d tests failed' % num_failed)
    for fail in tests_failed:
        print(fail)
    pass
