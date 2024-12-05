LDFLAGS  := -lm
EXES     := pthread

alls: $(EXES)

clean:
	rm -f $(EXES)

pthread: pthread.cc
	g++ -O3 -pthread -lm  pthread.cc  -o pthread
