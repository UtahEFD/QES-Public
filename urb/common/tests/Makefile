
KRNLDIR = ../
KRNLSRC = $(KRNLDIR)sum.cu \
		  		$(KRNLDIR)zero.cu \
		  		$(KRNLDIR)abs_diff.cu \
		  		$(KRNLDIR)pack.cu \
		  		$(KRNLDIR)max.cu \
		  		../../showerror.cu

KRNLOBJS = $(KRNLSRC:.cu=.o)

all: packTester zeroTester maxTester sumTester $(KRNLOBJS)

#packTester: packTester.cu $(KRNLOBJS)
#	nvcc $(KRNLOBJS) $< -o $@

#zeroTester: zeroTester.cu $(KRNLOBJS)
#	nvcc $(KRNLOBJS) $< -o $@

#maxTester: maxTester.cu $(KRNLOBJS)
#	nvcc $(KRNLOBJS) $< -o $@

sumTester: sumTester.cu $(KRNLOBJS)
	nvcc $(KRNLOBJS) $< -o $@


#absdifTester: absdifTester.cu $(KRNLOBJS)
#	nvcc $(KRNLOBJS) $< -o $@

%.o : %.cu
	nvcc -c $< -o $@

clean:
	rm -f $(KRNLOBJS)
	rm -f packTester zeroTester maxTester sumTester
