import torch

def get_gpu_name():
	return torch.cuda.get_device_name(0)

def get_all_gpu_names():
	d = []
	for i in range(torch.cuda.device_count()):
		d.append(torch.cuda.get_device_name(i))
	return d

if __name__ == '__main__':
	print(get_all_gpu_names())