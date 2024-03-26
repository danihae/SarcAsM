from .cuda_test import get_gpu_name # type: ignore
from .__cpu_count__ import get_cpu_count
from .__md5sum__ import md5sum
import os
import time
import platform
import subprocess
if platform.system() != 'Linux':
    raise Exception  # this script is designed to use Linux bash commands. Please use Linux

def get_hostname():
	return os.popen(f'hostname').read()

def get_memory_per_node():
	return os.getenv('SLURM_MEM_PER_NODE')

def get_cpu_model_name():
	raw_info = os.popen(f'lscpu | grep -i \'Model name\'').read()
	raw_info = raw_info[raw_info.find(':') + 1:].strip()
	return raw_info

def get_info_file_header():
	header = ''
	header += f'Time: {time.ctime()}\n'
	header += f'Node name: {get_hostname()}'
	header += f'CPU Model: {get_cpu_model_name()}\n'
	header += f'Core count: {get_cpu_count()}\n'
	header += f'Memory allocation: {get_memory_per_node()}\n'
	header += f'GPU: {get_gpu_name()}\n'
	header += '------------------------\n'
	return header

def write_info_file(outfile_name, additional_info):
	with open(outfile_name, 'w') as f:
		f.write(f'Info file name: {outfile_name}\n')
		f.write(get_info_file_header())
		f.write(additional_info)

def zoom_in(input, width, height, x=None, y=None, output='crop.mp4'):
	if x is None or y is None: # center video
		os.system(f'ffmpeg -i \'{input}\' -y -filter:v "crop={width}:{height}" \'{output}\'')
	else:	
		os.system(f'ffmpeg -i \'{input}\' -y -filter:v "crop={width}:{height}:{x}:{y}" \'{output}\' 2>/dev/null')

def create_zoomed_in_comparison(file1, file2, width, height, x=None, y=None, output='compare.mp4'):
	temp_1 = 'temp_' + file1.split('/')[-1]
	temp_2 = 'temp_' + file2.split('/')[-1]
	zoom_in(file1, width, height, x, y, output=temp_1)
	zoom_in(file2, width, height, x, y, output=temp_2)
	vertically_stack_two_videos(temp_1, temp_2, output)
	os.system(f'rm \'{temp_1}\'')
	os.system(f'rm \'{temp_2}\'')

def vertically_stack_two_videos(video1, video2, output):
	try:
		subprocess.run(["ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except:
		raise Exception   # ffmpeg not detected. Please install ffmpeg
	ffmpeg_command = f'ffmpeg -y -i \'{video1}\' -i \'{video2}\' -filter_complex vstack=inputs=2 \'{output}\''
	os.system(ffmpeg_command)
	with open(output + '.info.txt', 'w') as f:
		f.write(f'{get_info_file_header()}\nOperation: two stack\nOutfile name: {output}\nVideos:\n{video1}\t{video2}\n')


def grid_four_videos(video1, video2, video3, video4, output):
	try:
		subprocess.run(["ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except:
		raise Exception   # ffmpeg not detected. Please install ffmpeg
	ffmpeg_command = f'ffmpeg -y \
		-i \'{video1}\' -i \'{video2}\' -i \'{video3}\' -i \'{video4}\' \
		-filter_complex \
		"[0:v][1:v]hstack=inputs=2[top]; \
		[2:v][3:v]hstack=inputs=2[bottom]; \
		[top][bottom]vstack=inputs=2[v]" \
		-map "[v]" \
		\'{output}\''
	os.system(f'{ffmpeg_command}')
	with open(output + '.info.txt', 'w') as f:
		f.write(f'{get_info_file_header()}\nOperation: four stack\nOutfile name: {output}\nVideos:\n{video1}\t{video2}\n{video3}\t{video4}\n')

def grid_six_videos(video1, video2, video3, video4, video5, video6, output):
	try:
		subprocess.run(["ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except:
		raise Exception   # ffmpeg not detected. Please install ffmpeg
	ffmpeg_command = f'ffmpeg -y -i \'{video1}\' -i \'{video2}\' \
		-i \'{video3}\' -i \'{video4}\' \
		-i \'{video5}\' -i \'{video6}\' \
		-filter_complex \
		"[0:v][1:v][2:v]hstack=inputs=3[top];\
		[3:v][4:v][5:v]hstack=inputs=3[bottom];\
		[top][bottom]vstack=inputs=2[v]" \
		-map "[v]" \
		\'{output}\''
	os.system(f'{ffmpeg_command}')
	with open(output + '.info.txt', 'w') as f:
		f.write(f'{get_info_file_header()}\nOperation: six stack\nOutfile name: {output}\nVideos:\n{video1}\t{video2}\n{video3}\t{video4}\n{video5}\t{video6}\n')




if __name__ == '__main__':
	pass