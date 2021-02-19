
#Makefile

CC = nvcc

CFLAGS = -g -O3

ADD_SRCS =	

MAIN_SRCS = main.cu

INC_FILES = hist-equ.h

LIB_SRCS =	contrast_enhancement.cu

LIB_OBJS =	contrast_enhancement.o

LIB_NAME =	histogram

LIB_DIR = .

EXEC_NAME = main

all: $(EXEC_NAME)

$(EXEC_NAME):	$(MAIN_SRCS)	$(INC_FILES)	 $(LIB_NAME)
	$(CC) $(CFLAGS) $(MAIN_SRCS)	$(ADD_SRCS) -o $(EXEC_NAME)	-l$(LIB_NAME) -L$(LIB_DIR)
	
$(LIB_NAME): $(LIB_SRCS) $(INC_FILES)
	$(CC) -c $(CFLAGS) $(LIB_SRCS)
	ar rcsv lib$(LIB_NAME).a $(LIB_OBJS)

clean:
	rm -v $(EXEC_NAME) lib$(LIB_NAME).a *.o *.pgm