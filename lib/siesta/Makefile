
include arch.inc
sinclude libs.inc

OBJS = \
  m_io.o \
  m_siesta_hsx.o \
  m_precision.o \
  m_die.o \
  m_color.o

all: $(OBJS)
	gfortran -fPIC --shared $(OBJS) -lblas -o m_siesta_hsx.so

clean:
	rm $(OBJS)

%.o:%.F90 arch.inc
	$(TIME) $(FORTRAN) $(FLAGS) $(INCDIR) -c $<

