import os
import subprocess

try:
    subprocess.call(["md5sum", "--help"], stdout=subprocess.DEVNULL)
except FileNotFoundError:
	raise Exception # md5sum not found. Are you using Linux?

def md5sum(filename):
	"""
		returns the md5sum of a file, along with its filename, e.g.:
		0df61fe4ddf4455ba4d4e3c15abfabe2  predict_draft.py
	"""
	return os.popen(f'md5sum \'{filename}\'').read().split(' ')[0]

def md5sum_folder(folder_name):
	"""
		returns the md5sum of a file using the command `tar -cf - {folder_name} | md5sum`
	"""
	
	return os.popen(f'tar -cf - \'{folder_name}\' | md5sum').read()

if __name__ == '__main__':
	print(md5sum('predict_draft.py'))