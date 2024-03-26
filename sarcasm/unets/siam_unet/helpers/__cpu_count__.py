import multiprocessing

def get_cpu_count():
	return multiprocessing.cpu_count()	

if __name__ == '__main__':
	print('CPU count: ' +  str(get_cpu_count()))
