all : hello_gpu hello_gpu2

hello_gpu : hello_gpu.cu
	nvcc -o $@ $< 

hello_gpu2 : hello_gpu2.cu
	nvcc -o $@ $< 

clean :
	rm -f hello_gpu hello_gpu2
