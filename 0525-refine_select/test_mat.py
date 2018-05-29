import matlab.engine

eng = matlab.engine.start_matlab()
eng.demo(nargout=0)
eng.quit()
print('finish')


# import subprocess, os
# os.chdir('/disk5/yangle/DAVIS/code/cat_test/matcode/')
# subprocess.Popen(['matlab','demo.m'])

